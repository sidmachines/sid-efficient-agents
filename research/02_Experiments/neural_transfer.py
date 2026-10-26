"""Small, explicitly synthetic neural continual-learning extension using NumPy."""
import csv,json,time,sys
from pathlib import Path
import numpy as np
from study import Memory,deep_bytes

ROOT=Path(__file__).resolve().parent.parent
CENTERS=np.array([[-1,-1],[-1,1],[1,-1],[1,1]],float)

def run(method,capacity,seed):
    rng=np.random.default_rng(seed+55000)
    stream=np.concatenate([rng.normal(CENTERS[c],.35,size=(400,2)) for c in range(4)])
    labels=np.repeat(np.arange(4),400)
    init=np.random.default_rng(seed+77000)
    w1=init.normal(0,.3,size=(2,32));b1=np.zeros(32)
    w2=init.normal(0,.15,size=(32,4));b2=np.zeros(4)
    test_rng=np.random.default_rng(90001)
    test=np.concatenate([test_rng.normal(CENTERS[c],.35,size=(1000,2)) for c in range(4)])
    target=np.repeat(np.arange(4),1000)
    mem=Memory('coverage' if method=='class_coverage' else method,capacity,seed+88000) if capacity else None
    replay_rng=np.random.default_rng(seed+99000)
    records=[];elapsed=0.;processed=0
    start=time.perf_counter()
    for t,(x,y) in enumerate(zip(stream,labels),1):
        # Memory items group on (class, 0); x is payload only. The supervised
        # class label is available at training time, never the test set.
        xs=[x];ys=[int(y)]
        if mem and mem.items:
            for j in replay_rng.integers(len(mem.items),size=4):
                old=mem.items[int(j)];xs.append(old[2]);ys.append(old[0])
        X=np.asarray(xs);Y=np.asarray(ys);batch=len(Y);processed+=batch
        h=np.tanh(X@w1+b1)
        logits=h@w2+b2;logits-=logits.max(1,keepdims=True)
        probs=np.exp(logits);probs/=probs.sum(1,keepdims=True)
        probs[np.arange(batch),Y]-=1;probs/=batch
        gw2=h.T@probs;gb2=probs.sum(0)
        dh=(probs@w2.T)*(1-h*h)
        gw1=X.T@dh;gb1=dh.sum(0)
        w1-=.05*gw1;b1-=.05*gb1;w2-=.05*gw2;b2-=.05*gb2
        if mem:mem.add((int(y),0,tuple(map(float,x))))
        if t%400==0:
            elapsed+=time.perf_counter()-start
            predictions=(np.tanh(test@w1+b1)@w2+b2).argmax(1)
            acc=[float((predictions[target==c]==c).mean()) for c in range(4)]
            records.append(dict(method=method,capacity=capacity or 0,seed=seed,step=t,
                balanced_accuracy=float(np.mean(acc)),previous_classes_accuracy=float(np.mean(acc[:3])),
                class0=acc[0],class1=acc[1],class2=acc[2],class3=acc[3],processed_examples=processed,
                gradient_steps=t,train_seconds=elapsed,memory_bytes=deep_bytes(mem) if mem else 0))
            start=time.perf_counter()
    return records

def main():
    rows=[];start=time.perf_counter()
    configs=[('no_replay',None)]+[(m,c) for m in ['fifo','reservoir','class_coverage'] for c in [32,128]]
    for seed in range(20):
        for m,c in configs:rows.extend(run(m,c,seed))
        if (seed+1)%5==0:print(f'{(seed+1)*7}/140 neural runs complete',flush=True)
    with (ROOT/'03_Results/neural_runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    meta=dict(runs=140,real_training_examples=224000,wall_seconds=time.perf_counter()-start,
        python=sys.version,numpy=np.__version__,parameters=228,protocol=json.loads((Path(__file__).parent/'neural_protocol.json').read_text()))
    (ROOT/'03_Results/neural_metadata.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':main()
