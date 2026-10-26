"""Verify the complete evidence package against its recorded protocol."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
from study import specifications,Environment,Agent,deep_bytes
ROOT=Path(__file__).resolve().parent.parent

def main():
    base=ROOT/'03_Results'
    rows=list(csv.DictReader((base/'runs.csv').open()))
    meta=json.loads((base/'runs_metadata.json').read_text())
    assert len(rows)==21000
    assert meta['completed_runs']==2100
    assert hashlib.sha256((Path(__file__).parent/'study.py').read_bytes()).hexdigest()==meta['source_sha256']
    assert hashlib.sha256((Path(__file__).parent/'protocol.json').read_bytes()).hexdigest()==meta['protocol_sha256']
    keys=set()
    for r in rows:
        s=float(r['success']);assert math.isfinite(s) and -.0000001<=s<=1.0000001
        assert math.isfinite(float(r['return_value']))
        assert float(r['train_seconds'])>=0
        if int(r['capacity']):assert int(r['retained'])<=int(r['capacity'])
        if int(r['step'])==3000:
            keys.add((r['environment'],r['method'],int(r['capacity']) or None,int(r['seed']),int(r['sweeps']),int(r['every']),r['family']))
            if r['method']=='q_learning': expected=3000
            elif r['method']=='q_replay':expected=15000
            else:
                e=Environment(r['environment']);expected=(3000//int(r['every']))*int(r['sweeps'])*e.n*e.a
            assert int(r['backups'])==expected
    assert keys==set(specifications())
    metrics=list(csv.DictReader((base/'run_metrics.csv').open()))
    assert len(metrics)==2100
    for r in metrics:
        if r['method'] not in ['q_learning','q_replay']:
            e=Environment(r['environment'])
            assert int(float(r['array_base_correction_bytes']))==112+8*e.n*e.a
    # Confirm the accounting correction against the actual planner array base.
    a=Agent(21,4,'coverage',128,0,every=1)
    a.observe((0,0,-.01,1,False))
    import sys
    assert a.q.base is not None
    assert sys.getsizeof(a.q.base)==112+8*21*4
    neural=list(csv.DictReader((base/'neural_runs.csv').open()))
    assert len(neural)==560
    finals=[r for r in neural if int(r['step'])==1600]
    assert len(finals)==140
    for r in finals:
        assert 0<=float(r['balanced_accuracy'])<=1
        assert int(r['gradient_steps'])==1600
        assert int(r['processed_examples'])==(1600 if r['method']=='no_replay' else 7996)
    findings=json.loads((base/'findings.json').read_text())
    primary=[]
    for seed in range(20):
        ds=[]
        for env in ['grid_small','grid_large','grid_noisy','tool_workflow']:
            selected={r['method']:float(r['success_auc']) for r in metrics if r['environment']==env and int(float(r['seed']))==seed and int(float(r['capacity']))==128 and r['family']=='main'}
            ds.append(selected['coverage']-selected['fifo'])
        primary.append(np.mean(ds))
    assert abs(np.mean(primary)-findings['primary_difference'][0])<1e-12
    result={'status':'PASS','main_runs':2100,'neural_runs':140,'raw_main_checkpoints':21000,'raw_neural_checkpoints':560,'training_interactions':6300000,'primary_effect_recomputed':float(np.mean(primary)),
        'checks':['complete registered run matrix','source and protocol hashes','finite metrics and success bounds','buffer capacity','expected computation counts','array memory correction','neural training counts','primary effect recalculated from per-run data']}
    (base/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
