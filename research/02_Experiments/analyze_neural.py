"""Summarize the exploratory neural extension without pooling it into RL."""
import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
rng=np.random.default_rng(8621)
def interval(x):
    x=np.asarray(x,float)
    boot=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return [float(x.mean()),*map(float,np.percentile(boot,[2.5,97.5]))]
def main():
    rows=list(csv.DictReader((ROOT/'03_Results/neural_runs.csv').open()))
    last=[r for r in rows if r['step']=='1600']
    assert len(last)==140
    summaries=[]
    for method,cap in [('no_replay',0)]+[(m,c) for m in ['fifo','reservoir','class_coverage'] for c in [32,128]]:
        group=sorted([r for r in last if r['method']==method and int(r['capacity'])==cap],key=lambda r:int(r['seed']))
        s=dict(method=method,capacity=cap,n=20)
        for metric in ['balanced_accuracy','previous_classes_accuracy','train_seconds','memory_bytes','processed_examples']:
            s[metric]=interval([float(r[metric]) for r in group])
        summaries.append(s)
    a=np.array([float(r['balanced_accuracy']) for r in sorted(last,key=lambda r:int(r['seed'])) if r['method']=='class_coverage' and r['capacity']=='32'])
    b=np.array([float(r['balanced_accuracy']) for r in sorted(last,key=lambda r:int(r['seed'])) if r['method']=='fifo' and r['capacity']=='32'])
    result=dict(summary=summaries,coverage32_minus_fifo32=interval(a-b),note='Exploratory; paired training seeds; one fixed held-out synthetic test set; no model selection on test data')
    (ROOT/'03_Results/neural_findings.json').write_text(json.dumps(result,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.5))
    names={'no_replay':'No replay','fifo':'Recent memory','reservoir':'Reservoir','class_coverage':'Class coverage'}
    colors={'no_replay':'#9298A1','fifo':'#D35F51','reservoir':'#6E61A8','class_coverage':'#147D78'}
    subset=[r for r in summaries if r['capacity'] in [0,32]]
    x=np.arange(len(subset));means=np.array([s['balanced_accuracy'][0] for s in subset])
    low=np.array([s['balanced_accuracy'][1] for s in subset]);high=np.array([s['balanced_accuracy'][2] for s in subset])
    axes[0].bar(x,means,color=[colors[s['method']] for s in subset],yerr=[means-low,high-means],capsize=3)
    axes[0].set(xticks=x,xticklabels=[names[s['method']].replace(' ','\n') for s in subset],ylim=(0,1.05),ylabel='Final balanced test accuracy',title='Neural classification with 32 retained examples')
    for method in names:
        group=[r for r in rows if r['method']==method and int(r['capacity'])==(0 if method=='no_replay' else 32)]
        vals=[np.mean([float(r['class0']) for r in group if int(r['step'])==step]) for step in [400,800,1200,1600]]
        axes[1].plot([1,2,3,4],vals,marker='o',color=colors[method],label=names[method])
    axes[1].set(xlabel='Completed class block',ylabel='Test accuracy on the first class',xticks=[1,2,3,4],ylim=(0,1.05),title='Retention of earlier knowledge');axes[1].legend(fontsize=8,loc='lower left')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(ROOT/'04_Figures'/('05_neural_extension.'+ext),dpi=220,bbox_inches='tight')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
