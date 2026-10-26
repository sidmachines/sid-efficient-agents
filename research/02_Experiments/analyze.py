"""Analyze every registered run and produce tables and publication figures.

The primary aggregate bootstrap uses paired seed blocks across fixed tasks,
preserving shared-seed dependence. Secondary intervals are descriptive.
"""
import csv,json,sys,platform
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, NullFormatter

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'03_Results'
FIG=ROOT/'04_Figures'
NAMES={'q_learning':'Q learning','q_replay':'Q with replay','full_model':'Full model','fifo':'Recent memory','reservoir':'Reservoir','coverage':'Coverage memory'}
ENVS={'grid_small':'Small grid','grid_large':'Larger grid','grid_noisy':'Noisy grid','grid_shift':'Changed goal','tool_workflow':'Tool workflow'}
COLORS={'q_learning':'#9298A1','q_replay':'#A87526','full_model':'#202C45','fifo':'#D35F51','reservoir':'#6E61A8','coverage':'#147D78'}
N_STATES={'grid_small':21,'grid_large':38,'grid_noisy':21,'grid_shift':21,'tool_workflow':16}
STATIONARY=['grid_small','grid_large','grid_noisy','tool_workflow']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#263244','text.color':'#263244','axes.edgecolor':'#BBC2CA','grid.alpha':.2,'savefig.facecolor':'white'})
rng=np.random.default_rng(20261026)

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def ci(x,reps=10000):
    x=np.asarray(x,float)
    means=x[rng.integers(len(x),size=(reps,len(x)))].mean(axis=1)
    return [float(x.mean()),*map(float,np.percentile(means,[2.5,97.5]))]

def save(name):
    plt.savefig(FIG/(name+'.png'),dpi=220,bbox_inches='tight')
    plt.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.close()

def main():
    raw=list(csv.DictReader((DATA/'runs.csv').open()))
    numerical=['capacity','seed','sweeps','every','step','phase','success','return_value','train_seconds','backups','plans','agent_bytes','retained','model_edges','model_pairs','retained_pairs']
    groups=defaultdict(list)
    for r in raw:
        for k in numerical:r[k]=float(r[k])
        key=tuple(r[k] for k in ['environment','method','capacity','seed','sweeps','every','family'])
        groups[key].append(r)
    assert len(groups)==2100, len(groups)
    assert len(raw)==21000, len(raw)
    metrics=[]
    for key,rows in groups.items():
        rows.sort(key=lambda x:x['step'])
        assert [r['step'] for r in rows]==[0,100,250,500,1000,1500,1750,2000,2500,3000]
        last=rows[-1]
        auc=float(np.trapezoid([r['success'] for r in rows],[r['step'] for r in rows])/3000)
        post=[r for r in rows if r['step']>=1750]
        post_auc=float(np.trapezoid([r['success'] for r in post],[r['step'] for r in post])/1250)
        # The raw recursive estimator counts ndarray view headers, but not their
        # owning one-dimensional bases. Planning q is exactly one such view.
        # On the measured NumPy/CPython runtime its base is 112 + 8*S*A bytes.
        correction=112+8*N_STATES[last['environment']]*(5 if last['environment']=='tool_workflow' else 4) if last['plans']>0 else 0
        metrics.append({**{k:last[k] for k in ['environment','method','capacity','seed','sweeps','every','family']},
            'success_auc':auc,'post_shift_auc_1750_3000':post_auc,'final_success':last['success'],
            'final_return':last['return_value'],'train_seconds':last['train_seconds'],'backups':last['backups'],
            'agent_bytes_corrected':last['agent_bytes']+correction,
            'array_base_correction_bytes':correction,'retained_pairs':last['retained_pairs'],
            'model_pairs':last['model_pairs'],
            'checkpoint_peak_bytes_corrected':max(r['agent_bytes']+(correction if r['plans']>0 else 0) for r in rows)})
    write_csv(DATA/'run_metrics.csv',metrics)
    by_config=defaultdict(list)
    for r in metrics:
        key=tuple(r[k] for k in ['environment','method','capacity','sweeps','every','family'])
        by_config[key].append(r)
    summary=[]
    for key,records in by_config.items():
        assert len(records)==20
        r=dict(zip(['environment','method','capacity','sweeps','every','family'],key));r['n']=20
        for metric in ['success_auc','post_shift_auc_1750_3000','final_success','final_return','train_seconds','agent_bytes_corrected','backups','retained_pairs']:
            mean,lo,hi=ci([x[metric] for x in records])
            r.update({metric+'_mean':mean,metric+'_low':lo,metric+'_high':hi})
        summary.append(r)
    write_csv(DATA/'summary_by_environment.csv',summary)
    def select(env,method,cap=128,sweeps=20,every=50):
        if method in ['q_learning','full_model']:cap=0
        return sorted([r for r in metrics if r['environment']==env and r['method']==method and r['capacity']==cap and r['sweeps']==sweeps and r['every']==every],key=lambda x:x['seed'])
    def vector(env,method,metric,cap=128,sweeps=20,every=50):
        return np.array([r[metric] for r in select(env,method,cap,sweeps,every)])
    agg={}
    for method in NAMES:
        agg[method]={}
        for metric in ['success_auc','final_success','final_return','train_seconds','agent_bytes_corrected','backups','retained_pairs']:
            v=np.array([vector(env,method,metric) for env in STATIONARY]).mean(axis=0)
            agg[method][metric]=ci(v)
    diff=np.array([vector(env,'coverage','success_auc')-vector(env,'fifo','success_auc') for env in STATIONARY]).mean(axis=0)
    primary=ci(diff)
    effects=[]
    for env in ENVS:
        for cap in [32,64,128,256]:
            for other in ['fifo','reservoir']:
                d=vector(env,'coverage','success_auc',cap)-vector(env,other,'success_auc',cap)
                mean,lo,hi=ci(d)
                effects.append(dict(environment=env,capacity=cap,comparison='coverage minus '+other,metric='success_auc',mean=mean,low=lo,high=hi))
    write_csv(DATA/'paired_effects.csv',effects)
    main_curves={}
    for method in NAMES:
        vals=[]
        for env in STATIONARY:
            cap=0 if method in ['q_learning','full_model'] else 128
            g=[rows for k,rows in groups.items() if k[0]==env and k[1]==method and k[2]==cap and k[-1]=='main']
            g.sort(key=lambda rows:rows[0]['seed'])
            vals.append([[r['success'] for r in rows] for rows in g])
        main_curves[method]=np.array(vals).mean(axis=0)
    fig,ax=plt.subplots(figsize=(8.7,4.5))
    steps=np.array([0,100,250,500,1000,1500,1750,2000,2500,3000])
    for method,vals in main_curves.items():
        means=vals.mean(0)
        samples=vals[rng.integers(20,size=(10000,20))].mean(1)
        lo,hi=np.percentile(samples,[2.5,97.5],axis=0)
        ax.plot(steps,means,label=NAMES[method],color=COLORS[method],lw=2)
        ax.fill_between(steps,lo,hi,color=COLORS[method],alpha=.10)
    ax.set(xlabel='Real environment interactions',ylabel='Probability of success within 100 steps',ylim=(-.02,1.04),title='Learning efficiency across four stationary environments')
    ax.grid(True);ax.legend(ncol=3,loc='lower right',fontsize=9)
    fig.tight_layout();save('01_learning_curves')
    fig,axes=plt.subplots(1,4,figsize=(12.4,3.3),sharey=True)
    for ax,env in zip(axes,STATIONARY):
        for method in ['fifo','reservoir','coverage']:
            triples=np.array([ci(vector(env,method,'success_auc',cap)) for cap in [32,64,128,256]])
            ax.errorbar([32,64,128,256],triples[:,0],yerr=[triples[:,0]-triples[:,1],triples[:,2]-triples[:,0]],marker='o',capsize=3,color=COLORS[method],label=NAMES[method])
        ax.axhline(vector(env,'full_model','success_auc').mean(),color=COLORS['full_model'],ls='--',lw=1)
        ax.set(xscale='log',xticks=[32,64,128,256],xticklabels=['32','64','128','256'],title=ENVS[env],xlabel='Retained transitions',ylim=(-.02,1.04));ax.grid(True)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel('Success learning curve area')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,-.05))
    fig.tight_layout();save('02_memory_capacity')
    fig,axes=plt.subplots(1,2,figsize=(9.3,3.8))
    for method in NAMES:
        cap=0 if method in ['q_learning','full_model'] else 128
        g=[rows for k,rows in groups.items() if k[0]=='grid_shift' and k[1]==method and k[2]==cap and k[-1]=='main']
        vals=np.array([[r['success'] for r in rows] for rows in g])
        # Split at the change: avoid drawing a line across different objectives.
        for idx in [np.arange(6),np.arange(6,10)]:
            axes[0].plot(steps[idx],vals.mean(0)[idx],color=COLORS[method],label=NAMES[method] if idx[0]==0 else None,lw=1.7)
        x=agg[method]['agent_bytes_corrected'][0]/1024
        y=agg[method]['success_auc'][0]
        axes[1].scatter(x,y,color=COLORS[method],s=50)
        offsets={'q_learning':(4,8),'q_replay':(-20,-18),'full_model':(-30,7),'fifo':(-10,5),'reservoir':(4,-14),'coverage':(-30,12)}
        axes[1].annotate(NAMES[method],(x,y),xytext=offsets[method],textcoords='offset points',fontsize=8)
    axes[0].axvline(1500,color='#77818B',ls='--');axes[0].set(xlabel='Real environment interactions',ylabel='Success probability',title='Adaptation after the goal changes',ylim=(-.02,1.04));axes[0].legend(fontsize=7,ncol=2,loc='lower right');axes[0].grid(True)
    axes[1].set(xlabel='Estimated retained agent state (KiB)',ylabel='Stationary success curve area',title='Retained records are not total memory',ylim=(-.02,1.04));axes[1].grid(True);axes[1].margins(x=.25)
    fig.tight_layout();save('03_adaptation_and_memory')
    fig,axes=plt.subplots(1,2,figsize=(9.3,3.6))
    ablations={}
    for method in ['fifo','reservoir','coverage']:
        ablations[method]=[]
        for sweeps,every,label in [(20,50,'20 / 50'),(5,50,'5 / 50'),(20,200,'20 / 200')]:
            a=np.array([vector(env,method,'success_auc',128,sweeps,every) for env in STATIONARY]).mean(0)
            b=np.array([vector(env,method,'backups',128,sweeps,every) for env in STATIONARY]).mean(0)
            t=np.array([vector(env,method,'train_seconds',128,sweeps,every) for env in STATIONARY]).mean(0)
            mean,lo,hi=ci(a)
            ablations[method].append(dict(sweeps=sweeps,every=every,auc=[mean,lo,hi],backups=float(b.mean()),seconds=float(t.mean())))
            axes[0].errorbar(b.mean(),mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=COLORS[method],capsize=3)
            offset=(4,-14) if method=='fifo' and every==50 and sweeps==20 else (4,5)
            axes[0].annotate(label,(b.mean(),mean),xytext=offset,textcoords='offset points',fontsize=7,color=COLORS[method])
        axes[0].plot([],[],color=COLORS[method],label=NAMES[method])
    methods=list(NAMES)
    means=[agg[m]['train_seconds'][0] for m in methods]
    axes[1].barh([NAMES[m] for m in methods],means,color=[COLORS[m] for m in methods])
    axes[1].set(xlabel='Training seconds per run',title='Measured CPU training cost')
    axes[0].set(xlabel='Scalar action-value backups',ylabel='Stationary success curve area',title='Planning budget sensitivity at capacity 128');axes[0].legend(fontsize=7);axes[0].grid(True)
    fig.tight_layout();save('04_planning_cost')
    finding=dict(primary_difference=primary,stationary_aggregate=agg,planning_ablations=ablations,
        shifted={m:{metric:ci(vector('grid_shift',m,metric)) for metric in ['post_shift_auc_1750_3000','final_success','final_return']} for m in NAMES},
        main_runs=1500,ablation_runs=600,total_runs=2100,total_real_interactions=6300000,
        primary_bootstrap='paired seed-block bootstrap across four fixed stationary tasks; 10000 replicates',
        memory_correction='Added the exact owning ndarray base allocation omitted by recursive traversal: 112 + 8*S*A bytes for model agents after their first plan. This is CPython object-size accounting, not process RSS.',
        analysis_python=sys.version,analysis_numpy=np.__version__,analysis_matplotlib=matplotlib.__version__)
    (DATA/'findings.json').write_text(json.dumps(finding,indent=2))
    print(json.dumps(finding,indent=2))

if __name__=='__main__':main()
