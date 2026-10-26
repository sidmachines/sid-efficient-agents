"""Reproducible CPU study. Run: python study.py --workers 2

Only the evaluator sees the true transition kernel. Agents receive integer
state/action identifiers and observed (s,a,r,s_next,terminal) transitions.
Independent runs are processes, not collaborating AI agents.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, sys, time
from contextlib import nullcontext
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = json.loads((Path(__file__).parent / 'protocol.json').read_text())


class Environment:
    def __init__(self, name, seed=0, phase=0):
        self.name, self.rng = name, np.random.default_rng(seed)
        self.phase = phase
        if name == 'tool_workflow':
            self.n, self.a, self.start = 16, 5, 0
            self.dst = np.zeros((16, 5, 2), dtype=int)
            self.prob = np.zeros((16, 5, 2))
            self.reward = np.zeros((16, 5, 2))
            self.done = np.zeros((16, 5, 2), dtype=bool)
            for s in range(16):
                for a in range(5):
                    valid = a < 2 or (a == 2 and s & 3 == 3) or (a == 3 and bool(s & 4)) or (a == 4 and bool(s & 8))
                    nxt = s | (1 << a) if valid and a < 4 else s
                    terminal = valid and a == 4
                    self.dst[s,a] = [nxt, s]
                    self.prob[s,a] = [.85,.15]
                    self.reward[s,a] = [1 if terminal else (-.01 if valid else -.05), -.01]
                    self.done[s,a,0] = terminal
            self.cells = None
        else:
            large = name == 'grid_large'
            size = 7 if large else 5
            walls = {(1,1),(1,2),(2,2),(3,3)}
            if large:
                walls = {(1,1),(1,2),(1,3),(2,3),(3,1),(3,3),(3,5),(4,5),(5,1),(5,2),(5,3)}
            self.cells = [(r,c) for r in range(size) for c in range(size) if (r,c) not in walls]
            ix = {c:i for i,c in enumerate(self.cells)}
            self.n, self.a, self.start = len(ix), 4, ix[(0,0)]
            goal = (size-1,0) if name == 'grid_shift' and phase else (size-1,size-1)
            slip = .3 if name == 'grid_noisy' else .1
            self.dst = np.zeros((self.n,4,4), dtype=int)
            self.prob = np.zeros((self.n,4,4))
            self.reward = np.zeros((self.n,4,4))
            self.done = np.zeros((self.n,4,4), dtype=bool)
            for s,(r,c) in enumerate(self.cells):
                for a in range(4):
                    for k,(dr,dc) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
                        cell = (r+dr,c+dc)
                        sp = ix.get(cell,s)
                        terminal = self.cells[sp] == goal
                        self.dst[s,a,k] = sp
                        self.prob[s,a,k] = slip/4 + ((1-slip) if k == a else 0)
                        self.reward[s,a,k] = 1 if terminal else -.01
                        self.done[s,a,k] = terminal
        self.cdf = self.prob.cumsum(axis=2)
        self.state = self.start

    def step(self, action):
        s = self.state
        k = int(np.searchsorted(self.cdf[s,action], self.rng.random(), side='right'))
        k = min(k, self.dst.shape[2]-1)
        sp = int(self.dst[s,action,k])
        r, done = float(self.reward[s,action,k]), bool(self.done[s,action,k])
        self.state = sp
        return sp,r,done

    def reset(self):
        self.state = self.start
        return self.state

    def evaluate(self, policy, horizon=100):
        """Exact expected success and undiscounted return from the start."""
        ix = np.arange(self.n)
        dst, p = self.dst[ix,policy], self.prob[ix,policy]
        d, r = self.done[ix,policy], self.reward[ix,policy]
        success, ret = np.zeros(self.n), np.zeros(self.n)
        cont = p * (~d)
        immediate_success = (p*d).sum(1)
        immediate_reward = (p*r).sum(1)
        for _ in range(horizon):
            success = immediate_success + (cont*success[dst]).sum(1)
            ret = immediate_reward + (cont*ret[dst]).sum(1)
        return float(success[self.start]), float(ret[self.start])


class Memory:
    """Bounded experience retention. Coverage uses state-action groups only.

    Coverage: tentatively append; evict the oldest item in a group with the
    greatest occupancy. Ties use the oldest eligible item. No reward/goal oracle.
    Reservoir: Algorithm R, uniform sample of the complete observed stream.
    """
    def __init__(self, kind, capacity, seed):
        self.kind, self.capacity = kind, capacity
        self.rng = np.random.default_rng(seed)
        self.items = []
        self.seen = 0
        self.groups = Counter()

    def add(self, item):
        self.seen += 1
        if self.kind == 'reservoir' and len(self.items) == self.capacity:
            j = int(self.rng.integers(self.seen))
            if j >= self.capacity:
                return None, None
            old = self.items[j]
            self.items[j] = item
        else:
            self.items.append(item)
            self.groups[item[:2]] += 1
            old = None
            if len(self.items) > self.capacity:
                if self.kind == 'coverage':
                    maximum = max(self.groups.values())
                    j = next(i for i,x in enumerate(self.items) if self.groups[x[:2]] == maximum)
                else:
                    j = 0
                old = self.items.pop(j)
                self.groups[old[:2]] -= 1
                if self.groups[old[:2]] == 0:
                    del self.groups[old[:2]]
            return item, old
        self.groups[old[:2]] -= 1
        if self.groups[old[:2]] == 0:
            del self.groups[old[:2]]
        self.groups[item[:2]] += 1
        return item, old


class Agent:
    def __init__(self, n, a, method, capacity, seed, sweeps=20, every=50):
        self.n, self.a, self.method = n,a,method
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((n,a))
        self.sweeps, self.every = sweeps,every
        self.gamma, self.epsilon, self.alpha = .95,.15,.1
        self.steps, self.backups, self.plans = 0,0,0
        self.memory = Memory('fifo' if method == 'q_replay' else method, capacity, seed+17381) if capacity else None
        # Sparse sufficient statistics, including terminal outcome in the key.
        self.counts = {}
        self.reward_sums = {}

    def act(self, s):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.a))
        row = self.q[s]
        return int(self.rng.choice(np.flatnonzero(row == row.max())))

    def q_update(self, x):
        s,a,r,sp,d = x
        target = r + self.gamma * (0 if d else self.q[sp].max())
        self.q[s,a] += self.alpha * (target-self.q[s,a])
        self.backups += 1

    def change_count(self, x, sign):
        s,a,r,sp,d = x
        key = (s,a,sp,d)
        self.counts[key] = self.counts.get(key,0)+sign
        self.reward_sums[key] = self.reward_sums.get(key,0.0)+sign*r
        if self.counts[key] == 0:
            del self.counts[key], self.reward_sums[key]

    def observe(self, x):
        self.steps += 1
        if self.method in ('q_learning','q_replay'):
            self.q_update(x)
            if self.memory:
                self.memory.add(x)
                for _ in range(4):
                    j = int(self.memory.rng.integers(len(self.memory.items)))
                    self.q_update(self.memory.items[j])
            return
        if self.memory:
            added, removed = self.memory.add(x)
            if added is not None:
                self.change_count(added,1)
            if removed is not None:
                self.change_count(removed,-1)
        else:
            self.change_count(x,1)
        if self.steps % self.every == 0:
            self.plan()

    def plan(self):
        self.plans += 1
        if not self.counts:
            return
        keys = list(self.counts)
        ka = np.asarray(keys,dtype=int)
        sa = ka[:,0]*self.a+ka[:,1]
        count = np.asarray([self.counts[k] for k in keys], dtype=float)
        denom = np.bincount(sa,weights=count,minlength=self.n*self.a)
        probability = count / denom[sa]
        expected_r = np.bincount(sa,weights=np.asarray([self.reward_sums[k] for k in keys])/denom[sa],minlength=self.n*self.a)
        # Recompute from zero: no unmeasured long-term information in warm starts.
        value = np.zeros(self.n)
        for _ in range(self.sweeps):
            future = probability * (~ka[:,3].astype(bool)) * value[ka[:,2]]
            self.q = (expected_r + self.gamma*np.bincount(sa,weights=future,minlength=self.n*self.a)).reshape(self.n,self.a)
            value = self.q.max(axis=1)
        self.backups += self.sweeps*self.n*self.a

    def policy(self):
        return self.q.argmax(axis=1)


def deep_bytes(obj, seen=None):
    """CPython owned object estimate, not process RSS or temporary work arrays."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    size = sys.getsizeof(obj)
    if isinstance(obj,dict):
        size += sum(deep_bytes(k,seen)+deep_bytes(v,seen) for k,v in obj.items())
    elif isinstance(obj,(list,tuple,set)):
        size += sum(deep_bytes(x,seen) for x in obj)
    elif hasattr(obj,'__dict__'):
        size += deep_bytes(obj.__dict__,seen)
    return size


def run_one(spec):
    env_name, method, capacity, seed, sweeps, every, family = spec
    env = Environment(env_name, seed+100000)
    agent = Agent(env.n,env.a,method,capacity,seed,sweeps,every)
    checkpoints = PROTOCOL['checkpoints']
    total_steps = PROTOCOL['training_steps']
    rows, elapsed, episode_steps = [],0.0,0
    def record(step):
        success,ret = env.evaluate(agent.policy())
        rows.append(dict(environment=env_name, method=method, capacity=capacity or 0,
            seed=seed,sweeps=sweeps,every=every,family=family,step=step,phase=env.phase,
            success=success,return_value=ret,train_seconds=elapsed,backups=agent.backups,
            plans=agent.plans,agent_bytes=deep_bytes(agent),
            retained=len(agent.memory.items) if agent.memory else 0,
            model_edges=len(agent.counts),model_pairs=len({k[:2] for k in agent.counts}),
            retained_pairs=len(agent.memory.groups) if agent.memory else 0))
    record(0)
    s = env.reset()
    start = time.perf_counter()
    for step in range(1,total_steps+1):
        if env_name == 'grid_shift' and step == total_steps//2+1:
            old_rng = env.rng
            env = Environment(env_name,phase=1)
            env.rng = old_rng
            s, episode_steps = env.reset(),0
        a = agent.act(s)
        sp,r,done = env.step(a)
        agent.observe((s,a,r,sp,done))
        s = sp
        episode_steps += 1
        if done or episode_steps >= 100:
            s,episode_steps = env.reset(),0
        if step in checkpoints:
            elapsed += time.perf_counter()-start
            record(step)
            start = time.perf_counter()
    return rows


def specifications(smoke=False):
    specs=[]
    for env in PROTOCOL['environments']:
        for seed in range(PROTOCOL['seeds']):
            for method,cap in [('q_learning',None),('q_replay',128),('full_model',None)]+[(m,c) for m in PROTOCOL['retention'] for c in PROTOCOL['capacities']]:
                specs.append((env,method,cap,seed,20,50,'main'))
            for ab in PROTOCOL['planning_ablations']:
                for method in PROTOCOL['retention']:
                    specs.append((env,method,128,seed,ab['sweeps'],ab['every'],'planning_ablation'))
    if smoke:
        return [('grid_small',m,c,943,20,50,'smoke') for m,c in [('q_learning',None),('q_replay',128),('full_model',None),('fifo',32),('reservoir',128),('coverage',128)]]
    return specs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    specs=specifications(args.smoke)
    out=ROOT/'03_Results'
    out.mkdir(exist_ok=True)
    prefix='smoke' if args.smoke else 'runs'
    path=out/(prefix+'.csv')
    meta=dict(protocol=PROTOCOL,python=sys.version,numpy=np.__version__,platform=platform.platform(),
        processor=platform.processor(),logical_cpus=os.cpu_count(),workers=args.workers,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((Path(__file__).parent/'protocol.json').read_bytes()).hexdigest(),
        started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        planned_runs=len(specs))
    (out/(prefix+'_metadata.json')).write_text(json.dumps(meta,indent=2))
    start=time.perf_counter()
    with path.open('w',newline='') as f:
        writer=None
        with (ProcessPoolExecutor(max_workers=args.workers) if args.workers>1 else nullcontext()) as pool:
            iterator=pool.map(run_one,specs,chunksize=1) if pool else map(run_one,specs)
            for i,rows in enumerate(iterator,1):
                if writer is None:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader()
                writer.writerows(rows); f.flush()
                if i%25==0 or i==len(specs):
                    print(f'{i}/{len(specs)} runs complete | {time.perf_counter()-start:.1f}s',flush=True)
    meta['completed_runs']=len(specs)
    meta['wall_seconds']=time.perf_counter()-start
    meta['completed_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
    (out/(prefix+'_metadata.json')).write_text(json.dumps(meta,indent=2))

if __name__ == '__main__':
    main()
