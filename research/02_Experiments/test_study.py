"""Meaningful invariant and evaluator tests; python -m unittest -v test_study."""
import unittest
from collections import Counter
import numpy as np
from study import Agent,Memory,Environment,run_one

class StudyTests(unittest.TestCase):
    def test_retention_and_counts(self):
        for kind in ['fifo','reservoir','coverage']:
            agent=Agent(9,4,kind,13,31)
            rng=np.random.default_rng(1)
            for i in range(500):
                x=(int(rng.integers(9)),int(rng.integers(4)),float(rng.choice([-.01,1.])),int(rng.integers(9)),bool(rng.integers(2)))
                agent.observe(x)
                expected=Counter((s,a,sp,d) for s,a,r,sp,d in agent.memory.items)
                self.assertEqual(agent.counts,dict(expected))
                self.assertEqual(agent.memory.groups,Counter(x[:2] for x in agent.memory.items))
                self.assertLessEqual(len(agent.memory.items),13)
                for k in expected:
                    total=sum(x[2] for x in agent.memory.items if (x[0],x[1],x[3],x[4])==k)
                    self.assertAlmostEqual(agent.reward_sums[k],total)

    def test_coverage_preserves_rare_pair(self):
        m=Memory('coverage',4,0)
        m.add((0,0,1,1,True))
        for _ in range(100):m.add((1,0,0,1,False))
        self.assertIn((0,0,1,1,True),m.items)

    def test_terminal_does_not_bootstrap(self):
        for kind in ['q_learning','full_model']:
            a=Agent(2,1,kind,None,0,every=1)
            a.q[1,0]=100
            a.observe((0,0,1.,1,True))
            self.assertAlmostEqual(a.q[0,0],.1 if kind=='q_learning' else 1.)

    def test_evaluator_closed_form(self):
        env=Environment('tool_workflow')
        policy=np.full(16,4,dtype=int)
        env.start=15
        success,ret=env.evaluate(policy,3)
        self.assertAlmostEqual(success,1-.15**3)
        expected=sum(.15**t*(.85-.15*.01) for t in range(3))
        self.assertAlmostEqual(ret,expected)

    def test_transition_probabilities_and_reachability(self):
        for name in ['grid_small','grid_large','grid_noisy','grid_shift','tool_workflow']:
            env=Environment(name)
            np.testing.assert_allclose(env.prob.sum(2),1)
            self.assertTrue(np.all((env.dst>=0)&(env.dst<env.n)))
            v=np.zeros(env.n)
            for _ in range(100):
                q=(env.prob*(env.reward+.95*(~env.done)*v[env.dst])).sum(2)
                v=q.max(1)
            success,_=env.evaluate(q.argmax(1))
            self.assertGreater(success,.99)

    def test_seed_reproducibility(self):
        spec=('grid_small','coverage',32,987,5,50,'test')
        a,b=run_one(spec),run_one(spec)
        for x,y in zip(a,b):
            for key in ['success','return_value','backups','retained_pairs']:
                self.assertEqual(x[key],y[key])

if __name__=='__main__': unittest.main()
