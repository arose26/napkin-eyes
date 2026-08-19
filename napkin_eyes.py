#!/usr/bin/env python3
"""napkin-eyes: what should a trading DQN see? Observation ablation on the frozen sim.

One file, four commands:
  selfcheck  GPU env == napkin-tape CPU reference; no-look-ahead on features;
             vectorized 3-step hand-check; one-batch overfit; reward==log-equity
             identity; greedy-eval determinism
  sweep      6 observation arms x 10 seeds, fixed DQN recipe (inherited from
             napkin-replay: buffer + target net + 3-step, no double-Q, ratio 1)
  eval       re-run greedy test evals from saved nets
  plot       hero: test equity curves per arm, IQM + bootstrap CI bars

Repo 2 of the napkin-trader series. Registered hypotheses in README.md.
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
TAPE = os.path.join(HERE, "..", "napkin-tape", "out", "tape")
OUT = os.path.join(HERE, "out")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
          "JPM", "V", "XOM", "LLY", "WMT", "COST", "UNH"]
CRYPTOS = ["X:BTCUSD", "X:ETHUSD", "X:SOLUSD"]
UNIVERSE = STOCKS + CRYPTOS

TRAIN_END, VAL_END = "2026-01-01", "2026-06-01"   # walk-forward, time-ordered
GAMMA, NSTEP, LR, BATCH = 0.99, 3, 2.5e-4, 256
BUF_CAP, WARMUP, SYNC_EVERY = 200_000, 10_000, 500
N_ENVS, EP_LEN, VEC_STEPS = 256, 64, 4_000        # ~1M transitions/run
EPS_START, EPS_END, EPS_FRAC = 1.0, 0.05, 0.3
NOTIONAL = 100_000.0                              # slippage sizing assumption
ACTIONS = torch.tensor([-1.0, 0.0, 1.0])          # short / flat / long

ARMS = ["raw10", "raw20", "raw60", "norm20", "indic", "raw20pos"]
SEEDS = 10


# ------------------------------------------------------------------- tape

def load_tape():
    """Aligned daily bulk bars -> dates plus open/close/dollar-volume [T, S]."""
    by_sym = {}
    for sym in UNIVERSE:
        p = os.path.join(TAPE, sym.replace(":", "_") + ".bulk.jsonl")
        by_sym[sym] = {r["date"]: r for r in map(json.loads, open(p))}
    dates = sorted(set.intersection(*(set(v) for v in by_sym.values())))
    o = np.array([[by_sym[s][d]["o"] for s in UNIVERSE] for d in dates], np.float64)
    c = np.array([[by_sym[s][d]["c"] for s in UNIVERSE] for d in dates], np.float64)
    v = np.array([[by_sym[s][d]["v"] for s in UNIVERSE] for d in dates], np.float64)
    dv = np.zeros_like(c)
    for t in range(len(dates)):                    # trailing 20-bar dollar volume
        lo = max(0, t - 19)
        dv[t] = (v[lo:t + 1] * c[lo:t + 1]).mean(0)
    return dates, o, c, dv


def cost_frac(o, dv, is_crypto):
    """Per-unit-exposure switching cost under ClawStreet's published formula:
    commission ($0.005/share stocks, 5 bps crypto) + slippage (notional/dv * 50bps)."""
    comm = np.where(is_crypto, 5e-4, 0.005 / o)
    slip = (NOTIONAL / np.maximum(dv, 1e3)) * 50e-4  # guard: yahoo can emit v=0 rows
    return comm + slip


class Market:
    """Tensorized market shared by env + eval. Time index t is a bar; a step
    t -> t+1 holds p_old over the close->open gap, trades to p_new at the open
    (cost on |dp|), and rides p_new open->close. Daily-rebalanced +-1x exposure."""

    def __init__(self):
        self.dates, o, c, dv = load_tape()
        is_crypto = np.array([s.startswith("X:") for s in UNIVERSE])
        self.gap = torch.tensor(o[1:] / c[:-1] - 1, dtype=torch.float32, device=DEV)
        self.day = torch.tensor(c[1:] / o[1:] - 1, dtype=torch.float32, device=DEV)
        self.cost = torch.tensor(cost_frac(o, dv, is_crypto)[1:], dtype=torch.float32, device=DEV)
        self.logret = np.zeros_like(c)
        self.logret[1:] = np.log(c[1:] / c[:-1])
        self.T, self.S = self.gap.shape[0], len(UNIVERSE)  # step t: bar t -> t+1
        self.t_train_end = sum(1 for d in self.dates if d < TRAIN_END)
        self.t_val_end = sum(1 for d in self.dates if d < VAL_END)

    def step_factor(self, t, s, p_old, p_new):
        """Equity multiplier for step t on symbols s (all torch, batched)."""
        gap, day, cost = self.gap[t, s], self.day[t, s], self.cost[t, s]
        return ((1 + p_old * gap)
                * (1 - (p_new - p_old).abs() * cost)
                * (1 + p_new * day))


# ------------------------------------------------------------------- features

def build_features(market, arm, rng_future=None):
    """[T_bars, S, d] float32. Uses ONLY bars <= t for row t. rng_future, if given,
    randomizes all bars > t_check first (the no-look-ahead selfcheck rebuilds with
    scrambled future and asserts rows <= t_check are bit-identical)."""
    lr = market.logret.copy()
    if rng_future is not None:
        t_check, rng = rng_future
        lr[t_check + 1:] = rng.normal(0, 0.02, lr[t_check + 1:].shape)
    T, S = lr.shape
    x = lr * 100.0

    def window(h):
        # row t holds returns t, t-1, ..., t-h+1 (newest first); rows < h stay zero
        f = np.zeros((T, S, h), np.float32)
        for t in range(h, T):
            f[t] = x[t - h + 1:t + 1][::-1].T
        return f

    if arm.startswith("raw"):
        h = int(arm[3:].replace("pos", ""))
        return window(h)
    if arm == "norm20":
        f = window(20)
        mu = np.zeros((T, S), np.float32)
        sd = np.ones((T, S), np.float32)
        for t in range(60, T):
            seg = x[t - 59:t + 1]
            mu[t], sd[t] = seg.mean(0), seg.std(0) + 1e-6
        return ((f - mu[:, :, None]) / sd[:, :, None]).astype(np.float32)
    if arm == "indic":
        f = np.zeros((T, S, 8), np.float32)
        c = np.exp(np.cumsum(lr, 0))               # normalized price path
        for t in range(60, T):
            seg = lr[t - 13:t + 1]
            up = np.clip(seg, 0, None).sum(0)
            dn = -np.clip(seg, None, 0).sum(0) + 1e-9
            f[t, :, 0] = up / (up + dn) - 0.5                          # RSI-ish
            ema12 = c[max(0, t - 11):t + 1].mean(0)
            ema26 = c[max(0, t - 25):t + 1].mean(0)
            f[t, :, 1] = (ema12 / ema26 - 1) * 100                     # MACD-ish
            f[t, :, 2] = (c[t] / c[t - 19:t + 1].mean(0) - 1) * 10     # SMA20 dist
            f[t, :, 3] = (c[t] / c[t - 49:t + 1].mean(0) - 1) * 10     # SMA50 dist
            f[t, :, 4] = x[t]                                          # 1-bar ret
            f[t, :, 5] = x[t - 4:t + 1].sum(0)                         # 5-bar ret
            f[t, :, 6] = x[t - 19:t + 1].sum(0)                        # 20-bar ret
            f[t, :, 7] = x[t - 19:t + 1].std(0)                        # 20-bar vol
        return f
    raise SystemExit(f"unknown arm {arm}")


def obs_dim(arm):
    return {"raw10": 10, "raw20": 20, "raw60": 60, "norm20": 20,
            "indic": 8, "raw20pos": 23}[arm]


def make_obs(feat, t, s, pos, arm):
    """feat rows index BARS; decision at bar t sees feat[t]. pos: [-1,0,1] tensor."""
    o = feat[t, s]
    if arm.endswith("pos"):
        onehot = torch.zeros(len(s), 3, device=DEV)
        onehot[torch.arange(len(s)), (pos + 1).long()] = 1.0
        o = torch.cat([o, onehot], 1)
    return o


# ------------------------------------------------------------------- DQN

class QNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 128), nn.ReLU(),
                               nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(self, x):
        return self.f(x)


class Replay:
    """GPU ring buffer (napkin-replay's, tensorized)."""

    def __init__(self, cap, d, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.g, self.cap = g, cap
        self.s = torch.zeros(cap, d, device=DEV)
        self.a = torch.zeros(cap, dtype=torch.long, device=DEV)
        self.r = torch.zeros(cap, device=DEV)
        self.s2 = torch.zeros(cap, d, device=DEV)
        self.m = torch.zeros(cap, device=DEV)      # n-step length (gamma^m boot)
        self.n, self.i = 0, 0

    def add(self, s, a, r, s2, m):
        k = s.shape[0]
        idx = (self.i + torch.arange(k, device=DEV)) % self.cap
        self.s[idx], self.a[idx], self.r[idx] = s, a, r
        self.s2[idx], self.m[idx] = s2, m
        self.i = (self.i + k) % self.cap
        self.n = min(self.n + k, self.cap)

    def sample(self, k):
        idx = torch.randint(0, self.n, (k,), generator=self.g).to(DEV)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.m[idx]


class VecNStep:
    """Vectorized 3-step for synchronized fixed-length episodes (no true
    termination on financial tape -- every cut bootstraps). Keeps a rolling
    window of the last n (s, a, r) frames across N_ENVS envs."""

    def __init__(self, n, gamma, n_envs, d):
        self.n, self.gamma = n, gamma
        self.s = torch.zeros(n, n_envs, d, device=DEV)
        self.a = torch.zeros(n, n_envs, dtype=torch.long, device=DEV)
        self.r = torch.zeros(n, n_envs, device=DEV)
        self.k = 0                                  # frames buffered

    def push(self, s, a, r, s_next):
        """Returns tuples (s, a, R, s_boot, m) ready for the buffer, or None."""
        j = self.k % self.n
        self.s[j], self.a[j], self.r[j] = s, a, r
        self.k += 1
        if self.k < self.n:
            return None
        j0 = self.k % self.n                        # oldest frame
        R = torch.zeros_like(r)
        for i in range(self.n):
            R += (self.gamma ** i) * self.r[(j0 + i) % self.n]
        return (self.s[j0].clone(), self.a[j0].clone(), R, s_next,
                torch.full_like(r, float(self.n)))

    def flush(self, s_final):
        """Episode end: emit the <n-step tails, all bootstrapped from s_final."""
        out = []
        start = max(0, self.k - self.n + 1)
        for t0 in range(start, self.k):
            j0 = t0 % self.n
            m = self.k - t0
            R = torch.zeros(self.r.shape[1], device=DEV)
            for i in range(m):
                R += (self.gamma ** i) * self.r[(j0 + i) % self.n]
            out.append((self.s[j0].clone(), self.a[j0].clone(), R, s_final,
                        torch.full((self.r.shape[1],), float(m), device=DEV)))
        self.k = 0
        return out


def td_loss(net, tgt_net, batch):
    s, a, r, s2, m = batch
    q = net(s).gather(1, a[:, None]).squeeze(1)
    with torch.no_grad():                           # no double-Q (nodouble tied full)
        q2 = tgt_net(s2).max(1).values
        target = r + (GAMMA ** m) * q2              # truncation -> always bootstrap
    return ((q - target) ** 2).mean()


def train(arm, seed, market=None, feat=None, quiet=False, vec_steps=VEC_STEPS):
    torch.manual_seed(seed)
    market = market or Market()
    if feat is None:
        feat = torch.tensor(build_features(market, arm), device=DEV)
    d = obs_dim(arm)
    net, tgt = QNet(d).to(DEV), QNet(d).to(DEV)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    buf = Replay(BUF_CAP, d, seed)
    nstep = VecNStep(NSTEP, GAMMA, N_ENVS, d)
    acts = ACTIONS.to(DEV)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)

    def reset():
        h = 61                                      # deepest obs window + margin
        s = torch.randint(0, market.S, (N_ENVS,), generator=g).to(DEV)
        t = torch.randint(h, market.t_train_end - EP_LEN - NSTEP,
                          (N_ENVS,), generator=g).to(DEV)
        return s, t, torch.zeros(N_ENVS, device=DEV)

    sym, t, pos = reset()
    updates, losses = 0, []
    for step in range(vec_steps):
        o = make_obs(feat, t, sym, pos, arm)
        eps = EPS_START + min(1.0, step / (vec_steps * EPS_FRAC)) * (EPS_END - EPS_START)
        greedy = net(o).argmax(1)
        rand = torch.randint(0, 3, (N_ENVS,), generator=g).to(DEV)
        a = torch.where(torch.rand(N_ENVS, generator=g).to(DEV) < eps, rand, greedy)
        p_new = acts[a]
        r = torch.log(market.step_factor(t, sym, pos, p_new))
        pos, t = p_new, t + 1
        o2 = make_obs(feat, t, sym, pos, arm)
        emitted = nstep.push(o, a, r, o2)
        if emitted:
            buf.add(*emitted)
        if (step + 1) % EP_LEN == 0:
            for tup in nstep.flush(o2):
                buf.add(*tup)
            sym, t, pos = reset()
        while buf.n >= WARMUP and updates * BATCH < (step + 1) * N_ENVS:
            loss = td_loss(net, tgt, buf.sample(BATCH))   # ratio 1: trained == added
            opt.zero_grad(); loss.backward(); opt.step()
            updates += 1
            if updates % SYNC_EVERY == 0:
                tgt.load_state_dict(net.state_dict())
            losses.append(loss.item())
        if not quiet and (step + 1) % 1000 == 0:
            print(f"  {arm} s{seed} step {step+1}/{vec_steps} updates {updates} "
                  f"loss {np.mean(losses[-200:]):.4f}", flush=True)
    return net, market, feat


@torch.no_grad()
def evaluate(net, market, feat, arm, t0, t1):
    """Greedy rollout per symbol over steps [t0, t1); equal-weight portfolio curve."""
    net.eval()
    S = market.S
    sym = torch.arange(S, device=DEV)
    pos = torch.zeros(S, device=DEV)
    eq = torch.ones(S, device=DEV)
    curves = []
    for t in range(t0, t1):
        o = make_obs(feat, torch.full((S,), t, device=DEV), sym, pos, arm)
        p_new = ACTIONS.to(DEV)[net(o).argmax(1)]
        eq = eq * market.step_factor(torch.full((S,), t, device=DEV), sym, pos, p_new)
        pos = p_new
        curves.append(eq.mean().item())
    net.train()
    return curves


# ------------------------------------------------------------------- sweep

def iqm(x):
    x = sorted(x)
    k = max(1, len(x) // 4)
    return float(np.mean(x[k:-k])) if len(x) > 2 * k else float(np.mean(x))


def sweep(arms=ARMS, seeds=SEEDS):
    os.makedirs(os.path.join(OUT, "sweep"), exist_ok=True)
    market = Market()
    for arm in arms:
        feat = torch.tensor(build_features(market, arm), device=DEV)
        for seed in range(seeds):
            f = os.path.join(OUT, "sweep", f"{arm}_{seed}.json")
            if os.path.exists(f):
                continue
            t0 = time.time()
            net, _, _ = train(arm, seed, market, feat, quiet=True)
            val = evaluate(net, market, feat, arm, market.t_train_end, market.t_val_end)
            test = evaluate(net, market, feat, arm, market.t_val_end, market.T)
            json.dump({"arm": arm, "seed": seed, "val_curve": val, "test_curve": test,
                       "val_return_pct": (val[-1] - 1) * 100,
                       "test_return_pct": (test[-1] - 1) * 100},
                      open(f, "w"))
            torch.save(net.state_dict(), f.replace(".json", ".pt"))
            print(f"{arm} seed {seed}: val {(val[-1]-1)*100:+.2f}% "
                  f"test {(test[-1]-1)*100:+.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    report()


def report():
    import glob
    print(f"\n{'arm':10} {'n':>2} {'val IQM':>8} {'test IQM':>9}  test per-seed")
    for arm in ARMS:
        rs = [json.load(open(f)) for f in
              sorted(glob.glob(os.path.join(OUT, "sweep", f"{arm}_*.json")))]
        if not rs:
            continue
        tv = [r["test_return_pct"] for r in rs]
        vv = [r["val_return_pct"] for r in rs]
        print(f"{arm:10} {len(rs):>2} {iqm(vv):>+8.2f} {iqm(tv):>+9.2f}  "
              f"{[round(x, 1) for x in sorted(tv)]}")


# ------------------------------------------------------------------- selfcheck

def selfcheck():
    market = Market()
    print(f"tape: {market.T + 1} bars x {market.S} symbols, "
          f"train<{TRAIN_END} ({market.t_train_end}), val<{VAL_END} "
          f"({market.t_val_end - market.t_train_end}), test ({market.T - market.t_val_end})")

    # 1. GPU env == napkin-tape CPU reference on a scripted daily-rebalance policy
    sys.path.insert(0, os.path.join(HERE, "..", "napkin-tape"))
    import napkin_tape as nt
    sym = "AAPL"
    si = UNIVERSE.index(sym)
    script = [(-1.0 if (k // 5) % 3 == 0 else (1.0 if (k // 3) % 2 else 0.0))
              for k in range(60)]
    t0 = market.t_train_end   # 60 scripted bars fit between here and tape end
    # GPU env
    eq, pos = 1.0, 0.0
    gpu_curve = []
    for k, p in enumerate(script):
        f = market.step_factor(torch.tensor([t0 + k], device=DEV),
                               torch.tensor([si], device=DEV),
                               torch.tensor([pos], device=DEV),
                               torch.tensor([p], device=DEV))
        eq *= f.item(); pos = p
        gpu_curve.append(eq)
    # napkin-tape reference (same bulk bars, same daily targets)
    rows = [json.loads(l) for l in open(os.path.join(TAPE, sym + ".bulk.jsonl"))]
    rows = [r for r in rows if r["date"] in set(market.dates)]
    tape1 = nt.Tape({sym: rows})
    calls = {"k": 0}

    def scripted(view):
        k = calls["k"]; calls["k"] += 1
        return {sym: script[k]} if k < len(script) else {}
    ref_curve, _ = nt.run_sim(tape1, scripted, cash=100_000.0, warmup=t0)
    ref = [x / 100_000.0 for x in ref_curve[:len(script)]]
    worst = max(abs(g - r) / r for g, r in zip(gpu_curve, ref))
    assert worst < 2e-3, f"GPU env vs napkin-tape reference: {worst:.2%} deviation"
    print(f"selfcheck 1/6: GPU env matches napkin-tape reference "
          f"(worst {worst:.4%} over {len(script)} scripted bars)")

    # 2. no look-ahead: scramble every future bar, features at t are bit-identical
    t_check = market.t_train_end
    for arm in ("raw20", "norm20", "indic"):
        a = build_features(market, arm)
        b = build_features(market, arm, rng_future=(t_check, np.random.default_rng(0)))
        assert (a[:t_check + 1] == b[:t_check + 1]).all(), f"{arm} leaks the future"
        assert not (a[t_check + 5:] == b[t_check + 5:]).all(), f"{arm} scramble no-op?"
    print("selfcheck 2/6: features are bit-identical under future scrambling")

    # 3. vectorized 3-step against hand-computed values
    vn = VecNStep(3, 0.5, 1, 1)
    mk = lambda x: torch.full((1, 1), float(x), device=DEV)
    r_seq = [1.0, 2.0, 4.0, 8.0]
    outs = []
    for i, r in enumerate(r_seq):
        e = vn.push(mk(i), torch.zeros(1, dtype=torch.long, device=DEV),
                    torch.tensor([r], device=DEV), mk(i + 1))
        if e:
            outs.append(e)
    outs += vn.flush(mk(99))
    Rs = [round(o[2].item(), 4) for o in outs]
    ms = [o[4].item() for o in outs]
    assert Rs == [1 + 0.5 * 2 + 0.25 * 4, 2 + 0.5 * 4 + 0.25 * 8, 4 + 0.5 * 8, 8.0], Rs
    assert ms == [3.0, 3.0, 2.0, 1.0], ms
    print("selfcheck 3/6: 3-step tuples match hand computation (values, m, flush)")

    # 4. one fixed batch is overfittable
    torch.manual_seed(0)
    net = QNet(8).to(DEV)
    tgt = QNet(8).to(DEV); tgt.load_state_dict(net.state_dict())
    s = torch.randn(32, 8, device=DEV)
    batch = (s, torch.randint(0, 3, (32,), device=DEV), torch.randn(32, device=DEV),
             torch.randn(32, 8, device=DEV), torch.full((32,), 3.0, device=DEV))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(2000):
        loss = td_loss(net, tgt, batch)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 1e-4, loss.item()
    print(f"selfcheck 4/6: one batch overfit to {loss.item():.2e}")

    # 5. reward identity: cumulative env reward == log(final equity)
    torch.manual_seed(1)
    feat = torch.tensor(build_features(market, "raw10"), device=DEV)
    sym_v = torch.randint(0, market.S, (16,)).to(DEV)
    t_v = torch.full((16,), 100, device=DEV)
    pos_v = torch.zeros(16, device=DEV)
    eq = torch.ones(16, device=DEV)
    tot_r = torch.zeros(16, device=DEV)
    for k in range(30):
        p = ACTIONS.to(DEV)[torch.randint(0, 3, (16,)).to(DEV)]
        f = market.step_factor(t_v, sym_v, pos_v, p)
        tot_r += torch.log(f); eq *= f; pos_v = p; t_v = t_v + 1
    assert torch.allclose(tot_r, torch.log(eq), atol=1e-5)
    print("selfcheck 5/6: cumulative reward == log(final equity)")

    # 6. greedy eval determinism
    net = QNet(10).to(DEV)
    c1 = evaluate(net, market, feat, "raw10", market.t_val_end, market.T)
    c2 = evaluate(net, market, feat, "raw10", market.t_val_end, market.T)
    assert c1 == c2
    print("selfcheck 6/6: greedy eval is deterministic")
    print("ALL SELFCHECKS PASS")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selfcheck"
    if cmd == "sweep":
        arms = [a for a in sys.argv[2:] if a in ARMS] or ARMS
        sweep(arms)
    elif cmd == "report":
        report()
    else:
        {"selfcheck": selfcheck}[cmd]()
