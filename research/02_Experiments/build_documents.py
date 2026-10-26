"""Build the paper and supplement from recorded results using ReportLab.
Editable HTML versions are generated from the same content as the PDFs.
"""
from pathlib import Path
import json,csv,html
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'01_Paper'
BRAND=json.loads((OUT/'branding.json').read_text(encoding='utf-8'))
COMPANY=BRAND['company']
F=json.loads((ROOT/'03_Results/findings.json').read_text())
N=json.loads((ROOT/'03_Results/neural_findings.json').read_text())
META=json.loads((ROOT/'03_Results/runs_metadata.json').read_text())
REF=json.loads((ROOT/'05_Research_Notes/references.json').read_text())
names={'q_learning':'Q learning','q_replay':'Q with replay','full_model':'Full model','fifo':'Recent memory','reservoir':'Reservoir','coverage':'Coverage memory'}
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='PaperTitle',fontName='Times-Bold',fontSize=23,leading=26,textColor=colors.black,spaceAfter=10))
styles.add(ParagraphStyle(name='PaperSub',fontName='Helvetica',fontSize=10,leading=14,textColor=colors.HexColor('#48515C'),spaceAfter=12))
styles.add(ParagraphStyle(name='BodyPaper',fontName='Times-Roman',fontSize=11,leading=14.4,spaceAfter=8))
styles.add(ParagraphStyle(name='HeadPaper',fontName='Helvetica-Bold',fontSize=14,leading=18,spaceBefore=8,spaceAfter=8,textColor=colors.black))
styles.add(ParagraphStyle(name='SubPaper',fontName='Helvetica-Bold',fontSize=11,leading=14,spaceBefore=7,spaceAfter=5,textColor=colors.black))
styles.add(ParagraphStyle(name='CaptionPaper',fontName='Helvetica',fontSize=8.5,leading=11,spaceBefore=4,spaceAfter=10,textColor=colors.HexColor('#424B56')))
styles.add(ParagraphStyle(name='CellPaper',fontName='Helvetica',fontSize=8.5,leading=11))
styles.add(ParagraphStyle(name='RefPaper',fontName='Times-Roman',fontSize=10,leading=12.7,spaceAfter=7))
styles.add(ParagraphStyle(name='CodePaper',fontName='Courier',fontSize=8,leading=11,spaceAfter=7))

def p(text):return ('p',text)
def h(text):return ('h',text)
def sub(text):return ('sub',text)
def fig(file,caption,width=500):return ('fig',file,caption,width)
def table(headers,rows,widths):return ('table',headers,rows,widths)
def pct(x):return f'{100*x:.1f}%'
def ci_text(x,scale=1,digits=3):
    return f'{scale*x[0]:.{digits}f} [{scale*x[1]:.{digits}f}, {scale*x[2]:.{digits}f}]'

paper=[]
paper.append([
 ('title','Memory Selection and Planning Budgets in Learning Agents'),
 ('subtitle','SID Machines | Private research study'),
 h('Abstract'),
 p('Efficient learning requires a choice about which experiences to retain and how much computation to spend reusing them. We extend a small gridworld project into a controlled study of 2,100 seeded agent runs across five finite-state environments, followed by 140 exploratory neural-classifier runs. Agents receive equal budgets of 3,000 real interactions. We compare Q learning, experience replay, planning from full transition statistics, and planning from recent, reservoir, or coverage-based memories. Policy success is evaluated exactly over a 100-step horizon. At the preselected 128-transition capacity, coverage memory improves the stationary success-curve area over recent memory by 0.0106, with a paired 95% bootstrap interval of [-0.0095, 0.0290]; this does not establish an average advantage. After a goal change, recent memory achieves a post-change success area of 0.971, compared with 0.472 for full-history statistics. In a separate synthetic neural task, 32-example class-balanced and reservoir memories retain about 99.3% balanced accuracy, versus 89.4% with recent memory. Limited raw-history capacity does not guarantee lower total storage or computation. The results support task-dependent memory selection and explicit planning budgets, while providing a testable design for future LLM-agent evaluation. No LLM or GPU experiment is reported.'),
 h('1 Introduction'),
 p('The starting question was whether an agent can remain effective while learning from less experience. A learned transition model can reuse one interaction in many planning updates, but this reuse has a computational cost. A short memory can remove obsolete evidence, yet it can also erase the only observations connecting a start state to a goal. These effects should be measured separately.'),
 p('This study asks how retention strategy and planning depth affect task success under fixed interaction budgets. Its contribution is an auditable comparison and a characterization of failure modes across stationary control, changed objectives, and a small neural continual-learning task. It does not introduce a new reinforcement-learning family or establish a universal memory threshold.'),
 sub('Relation to earlier work'),
 p('Dyna connects learned models, planning, and reinforcement learning [1]. Reservoir sampling supplies a standard history-sampling baseline [2]. Selective experience replay, including coverage and distribution matching, is established prior work [3]. Our experiment therefore tests a concrete selection rule rather than claiming novelty for selective memory. Reporting seed uncertainty follows the evaluation concerns raised by Agarwal et al. [4]; separating performance from cost follows Kapoor et al. [8].')
])

paper.append([
 h('2 Agents and memory mechanisms'),
 p('Every agent observes a state identifier, selects an action, and receives a reward, next-state identifier, and terminal flag. State and action counts are known, but transition probabilities, rewards, walls, and the goal location are not supplied. Training uses epsilon-greedy exploration with epsilon 0.15 and discount 0.95. Ties during training are randomized; evaluation uses the first maximizing action deterministically.'),
 table(['Method','Learning and retention rule'],[
 ['Q learning','One tabular temporal-difference update per real interaction; learning rate 0.1.'],
 ['Q with replay','The same update plus four uniform replay updates from a recent 128-transition buffer.'],
 ['Full model','All observed transition counts and reward sums; raw history is not stored.'],
 ['Recent memory','Model statistics computed from the most recent M transitions, with oldest-first eviction.'],
 ['Reservoir','Model statistics from a uniform sample of M transitions from all experience so far.'],
 ['Coverage memory','Preferentially remove old records from the most represented state-action group.']
 ],[104,396]),
 sub('Coverage retention rule'),
 p('After appending a transition, count buffer records by their state-action pair. If capacity is exceeded, identify the largest group count and evict the oldest record belonging to any group with that count. The rule encourages coverage while replacing redundant observations. It does not use future rewards, goal labels, the evaluator, or an importance oracle. With very small capacity, retaining broad coverage can still leave too few observations per transition distribution.'),
 sub('Planning from retained evidence'),
 p('Model-based agents estimate transitions from counts and expected rewards from reward sums. Terminal outcomes have zero continuation value. At each planning event, synchronous value iteration starts from zero and performs 20 sweeps by default. Each sweep evaluates every state-action pair using the learned expected immediate reward plus discounted successor value. Unobserved pairs have value zero. The final action-value table defines the greedy policy. Planning occurs every 50 interactions; evaluations do not trigger an additional plan.'),
 p('Resetting planning values to zero prevents warm-start values from silently retaining information removed from the memory buffer. The current policy remains in use between planning events, so the transition capacity is not a hard cap on every form of stored information. Agents also retain action values, counters, random-generator state, and sparse model statistics. We report estimated total retained agent-state bytes separately.'),
 sub('Implementation costs'),
 p('Coverage eviction scans a buffer of at most M records, giving O(M) eviction time in this implementation. Planning performs work proportional to the retained transition edges and the number of state-action pairs per sweep. Full-model storage grows with distinct observed outcomes, not with the length of the raw trajectory. These implementation details materially affect efficiency comparisons.')
])

agg=F['stationary_aggregate']
result_rows=[[names[m],f"{agg[m]['success_auc'][0]:.3f}",f"[{agg[m]['success_auc'][1]:.3f}, {agg[m]['success_auc'][2]:.3f}]",pct(agg[m]['final_success'][0]),f"{agg[m]['agent_bytes_corrected'][0]/1024:.1f}"] for m in names]
paper.append([
 h('3 Experimental design'),
 table(['Environment','States and dynamics','Purpose'],[
 ['Small grid','5 by 5; 21 states; 0.10 action slip','Connection to original project'],
 ['Larger grid','7 by 7; 38 states; 0.10 action slip','Longer paths and greater coverage demand'],
 ['Noisy grid','Small grid with 0.30 action slip','Transition-estimation sensitivity'],
 ['Changed goal','Small grid; goal moves after step 1500','Adaptation to obsolete reward evidence'],
 ['Tool workflow','16 encoded states; 6 reachable; 5 actions','Finite-state prerequisite and retry task']
 ],[90,220,190]),
 p('Grid slip chooses uniformly among all four actions, including the intended action. Walls cause no movement. Steps cost -0.01 and reaching the goal gives +1. The changed goal moves to the bottom-left corner without notifying the agent. The simulated tool task requires two fetches, a join, validation, and submission, with 15% transient failure. Full reward and transition rules are in the code.'),
 p('The main design contains 15 configurations per environment: three baselines and three memory rules at capacities 32, 64, 128, and 256. Twenty paired seed values produce 1,500 runs. A further 600 runs test 5 sweeps every 50 steps and 20 sweeps every 200 steps at capacity 128. Every run receives 3,000 interactions; episodes reset on success or after 100 steps. Methods act on their own policies and therefore do not receive identical experience streams.'),
 p('The primary endpoint is normalized success-curve area across four stationary environments; the primary contrast is coverage minus recent memory at capacity 128. Ten checkpoints span 0 to 3,000 interactions. Exact evaluation removes rollout noise, not uncertainty about unseen environments. Intervals resample 20 paired seed blocks 10,000 times. Secondary comparisons are exploratory and not multiplicity-adjusted.'),
 h('4 Main results'),
 table(['Method','Success area','95% interval','Final success','State KiB'],result_rows,[103,68,125,100,104]),
 p('Table 1. Means across four stationary environments at capacity 128 where applicable. Success area rewards early and sustained learning; it is not final accuracy. State KiB estimates retained Python objects and array storage, excluding process overhead and temporary planning arrays. Full model has no raw-transition capacity.'),
 p('Full-model planning learns fastest here. The primary coverage-minus-recent difference is compatible with a small disadvantage or advantage. Replay improves Q learning. Interaction budgets are equal; computation budgets differ.')
])

paper.append([
 h('5 Memory capacity changes the outcome'),
 fig('01_learning_curves.png','Figure 1. Stationary learning curves at capacity 128, averaged equally across four fixed environments. Shading shows pointwise 95% paired seed-block bootstrap intervals. Curves between checkpoints are visual interpolation.'),
 fig('02_memory_capacity.png','Figure 2. Success-curve area versus retained-transition capacity. Error bars are 95% seed-bootstrap intervals within each environment. Dashed lines show the full-model baseline. All capacities and outcomes are reported.'),
 p('The original project suggested a transition near 128 stored experiences in one map. The expanded study shows why that number cannot be transferred as a general threshold. At capacity 32, recent memory scores 0.979 success area on the short tool workflow but only 0.033 on the larger grid. Coverage scores 0.956 and 0.201 respectively. Larger buffers usually help, but the recent-memory curve is not monotonic on the larger grid. Capacity changes the learned policy and hence subsequent experience, so more history need not improve every finite-budget run.'),
 p('Coverage retains more distinct state-action pairs at capacity 128: a stationary mean of 70.1 versus 26.0 for recent memory. Greater breadth alone does not guarantee better control; local estimation quality, goal connectivity, and timely replacement also matter. The reservoir baseline performs poorly in the larger grid at this budget, despite being useful in the neural extension.')
])

paper.append([
 h('6 Changing objectives and actual memory cost'),
 fig('03_adaptation_and_memory.png','Figure 3. Left: mean success before and after the goal moves at interaction 1500. The line is deliberately broken across the objective change. Right: estimated final retained agent state versus stationary success area at capacity 128. Point estimates are descriptive; intervals are in the data tables.'),
 sub('Forgetting can improve adaptation'),
 p('On the changed-goal task, recent memory achieves a post-change success area of 0.971 [0.942, 0.993], versus 0.472 [0.278, 0.670] for full-model statistics and 0.334 [0.173, 0.504] for reservoir memory. Coverage reaches 0.960 [0.915, 0.990]. These areas cover the measured post-change interval from 1750 to 3000; they exclude the first 250 interactions after the change.'),
 p('The retained historical outcomes now mix two reward and termination regimes. Recent memory replaces obsolete evidence quickly; a reservoir continues to represent the entire observed history. This is a plausible explanation consistent with the design, rather than a separately isolated causal mechanism. The experiment tests a specific unannounced goal change. A model receiving an explicit task identifier or using recency-weighted statistics could behave differently.'),
 sub('A bounded buffer is not automatically smaller'),
 p('Full-model planning retains about 25.7 KiB on the stationary suite, compared with 31.2 KiB for recent memory and 44.0 KiB for coverage at capacity 128. Full-model counts aggregate repeated outcomes, whereas bounded agents retain individual transitions, group counts, and model statistics. Therefore, a buffer containing only 4.3% of the 3,000 observed transitions can still occupy more space than a full learned tabular model.'),
 p('The byte estimate recursively counts retained Python objects and NumPy storage. Analysis explicitly corrects the raw counter for the owning array behind the planner action-value view. This correction is exposed as its own column. The estimate is implementation-dependent and excludes temporary work arrays, process memory, and device allocations. It should not be read as a GPU-memory measurement.'),
 sub('Implication'),
 p('For compact tabular dynamics, aggregate statistics are a strong baseline. Raw-memory reduction is useful only if its storage representation, selection overhead, and adaptation behavior fit the application. A claim of overall efficiency must name the resource actually reduced.')
])

paper.append([
 h('7 Planning depth and frequency'),
 fig('04_planning_cost.png','Figure 4. Left: planning ablations at capacity 128. Labels give sweeps per plan / interactions between plans. The 5 / 50 and 20 / 200 settings use the same scalar-backup budget. Right: mean measured training time excluding evaluation and memory inspection, on this CPU implementation.'),
 p('At the same mean budget of 30,000 scalar action-value backups, coverage with 20 sweeps every 200 interactions obtains success area 0.880, versus 0.695 with 5 sweeps every 50 interactions. Recent memory shows a larger separation: 0.823 versus 0.385. Allocating the same number of updates across planning events can therefore produce different policies.'),
 p('Because planning restarts from zero, a synchronous backup propagates reward information only one transition farther per sweep. Five sweeps may be insufficient to connect a distant reward to the start state. This explains why cutting the number of sweeps is a different intervention from planning less often. The result is specific to this planner; warm starts, asynchronous updates, prioritized sweeping, and learned value functions are not tested.'),
 p('At the default setting, coverage takes about 0.252 seconds per stationary run, recent memory 0.145 seconds, and full-model planning 0.128 seconds. Q learning takes 0.114 seconds, while replay takes 0.234 seconds. Vectorized planning and Python-level replay have different per-backup costs. Scalar backup counts are therefore a work proxy, not FLOPs, energy, or a universal runtime conversion.'),
 p('The timing includes action selection, environment simulation, learning, memory selection, and planning. It excludes policy evaluation and deep memory inspection. Runs execute serially on a shared Windows CPU host; subsecond timings are descriptive and sensitive to implementation and host activity. No hardware energy or GPU utilization was measured.'),
 sub('Practical interpretation'),
 p('An agent designer should budget both real interactions and internal planning. If interactions are expensive, extra planning may be worthwhile. If the environment is cheap and the state space is small, a simpler learner may be preferable. A joint evaluation should report a success-cost curve, rather than treating either fewer interactions or fewer retained records as sufficient evidence of lower total cost.')
])

neural_rows=[]
for r in N['summary']:
    label={'no_replay':'No replay','fifo':'Recent memory','reservoir':'Reservoir','class_coverage':'Class coverage'}[r['method']]
    neural_rows.append([label,str(r['capacity']),ci_text(r['balanced_accuracy'],100,2),f"{r['processed_examples'][0]:,.0f}"])
paper.append([
 h('8 Exploratory neural extension'),
 p('To test whether the memory question is meaningful beyond tabular planning, a separate experiment trains a small neural classifier with 2 inputs, 32 tanh hidden units, 4 softmax outputs, and 228 parameters. Four Gaussian classes arrive in sequential blocks of 400 examples. Training uses one gradient step per arriving example; replay methods add four retained examples to each minibatch. A fixed, independently generated balanced test set contains 1,000 examples per class. Twenty initialization and data seeds across seven configurations produce 140 runs.'),
 fig('05_neural_extension.png','Figure 5. Synthetic neural continual learning. Left: final balanced test accuracy with capacity 32, with 95% paired-seed bootstrap intervals. Right: mean test accuracy on the first class as later classes arrive. The task is deliberately small and does not model language understanding.'),
 table(['Method','Capacity','Accuracy percent and 95% interval','Examples processed'],neural_rows,[102,57,229,112]),
 p('Class coverage groups examples by their observed training label and evicts from the most populated class. At capacity 32, class coverage exceeds recent memory by 9.90 percentage points [8.72, 11.10], while reservoir memory is similarly effective. The small numerical difference between class coverage and reservoir is not presented as a meaningful advantage.'),
 p('Unlike the changed-goal task, this evaluation still requires earlier classes to be recognized. A memory preserving historical diversity is therefore useful. All replay methods process 7,996 training-example presentations versus 1,600 without replay, although they use the same number of gradient steps. This is evidence of improved retention under replay, not evidence of reduced training compute. The extension is exploratory, uses one synthetic distribution, and does not establish transfer to deep networks or LLMs.')
])

paper.append([
 h('9 Implications for practical LLM agents'),
 p('The relevant LLM application is a memory and planning layer around a fixed pretrained model. It would store verified action outcomes, retrieve a bounded set relevant to a current task, and use those records when selecting or checking tool calls. ReAct establishes a reasoning-and-action loop [5], Reflexion uses feedback and memory without updating model weights [6], and LATS combines language agents with search [7]. These precedents make the application plausible; they do not validate the specific retention rule studied here.'),
 p('A candidate memory record would include an abstract task state, tool name, required conditions, observed outcome, success or failure, and environment version. Recent records would help replace obsolete tool behavior; a separate diverse archive could preserve rare but valid failure cases. In a language setting, the state abstraction itself can be wrong, and records vary in length. A fair test must impose token or byte budgets, measure retrieval and summarization costs, and validate outcomes independently. A hybrid archive is a proposal, not an evaluated method in this paper.'),
 p('A practical next experiment would compare the same frozen model and prompt with no cross-task memory, recent memory, reservoir memory, and coverage-based memory on held-out tool tasks. It should measure task success, invalid calls, input and output tokens, total latency, and adaptation after a version change. Calibration tasks must be separate from final test tasks, and memory must never contain hidden test answers. The supplement specifies a local evaluation plan with a prospective success criterion.'),
 p('Larger pretraining datasets are not inherently the cause of agent failures. Pretraining cost, inference cost, task adaptation, and long-context retrieval are different problems. Long-context retrieval limitations have been observed in specific settings [9], but the experiments here neither compress model weights nor reduce pretraining data. No GPU savings, LLM accuracy improvement, deployed reliability, or broad model-size claim follows from these results.'),
 h('10 Limitations'),
 p('The control tasks have exact state identifiers, small discrete action spaces, fixed starts, and known state/action counts. Three stationary tasks share a navigation family; the tool task has only six reachable encoded states. There is no held-out map distribution, partial observability, language parsing, or real API execution. Exact evaluation is exact only for the supplied simulator and 100-step horizon. The seed intervals condition on these selected tasks, not on all possible applications.'),
 p('Hyperparameters are fixed rather than exhaustively tuned for every baseline. Only three retention rules and two planning perturbations are tested. Dense action-value storage and sparse Python dictionaries affect memory and runtime. Planning restarts from zero, making sweep depth especially important. The neural extension uses small synthetic data and label-aware grouping; it should not be equated with unsupervised LLM memory. Secondary intervals are exploratory and unadjusted for multiple comparisons.'),
 h('11 Conclusion'),
 p('The experiments support a conditional answer to the original efficiency question. Reusing experience through a learned model can improve interaction efficiency, but memory selection must match whether past information remains useful. Coverage is not a universal winner, recent evidence can aid adaptation, and full transition statistics can be smaller than a bounded raw buffer. These distinctions provide concrete design and evaluation requirements for future practical agents.')
])

refs=[h('References')]
for r in REF:refs.append(('ref',f"[{r['id']}] {html.escape(r['citation'])}<br/><link href='{r['url']}' color='#145A70'>{r['url']}</link>"))
refs += [h('Reproducibility and provenance'),p(f"The archive includes all 21,000 main-study checkpoint records, 560 neural checkpoint records, per-run metrics, configuration summaries, source code, software versions, protocols, and an unchanged copy of the original project. Main execution completed {META['completed_runs']:,} runs in {META['wall_seconds']:.1f} seconds on the recorded host. These are seeded experimental repetitions, not 2,100 distinct agent architectures."),p('The primary local protocol was recorded before the main run; it was not publicly preregistered. A bootstrap specification and array-storage accounting correction are documented in the research notes. The neural extension has a separate prospective protocol. This study belongs to the SID Machines research project and builds on its original gridworld investigation. All new numerical results were executed locally. No external model API was called.')]
paper.append(refs)

supp=[]
supp.append([
 ('title','Technical Supplement and Reproduction Guide'),
 ('subtitle','Companion to Memory Selection and Planning Budgets in Learning Agents'),
 h('1 Reproduce the experiments'),
 p('Use Python 3.12 or later. Install the experiment requirements, then run the commands below from the 02_Experiments directory. The supplied results are already complete; rerunning study.py overwrites runs.csv and its metadata. Copy the package before experimenting if you want to preserve the original run files.'),
 ('code','python -m pip install -r requirements.txt<br/>python -m unittest -v test_study<br/>python study.py --workers 1<br/>python analyze.py<br/>python neural_transfer.py<br/>python analyze_neural.py<br/>python verify_results.py'),
 p('Training requires only NumPy. Analysis and charts require Matplotlib. The supplied agent results use Python 3.12.14 and NumPy 2.3.5; figures and bootstrap analysis were produced with Python 3.13, NumPy 2.3.3, and Matplotlib 3.10.6. Actual versions are saved in the metadata. A single consistent environment satisfying requirements.txt can run the full pipeline; tiny floating-point or runtime differences across systems are possible.'),
 p('The PDF and editable HTML sources are generated by build_documents.py using the saved findings. To rebuild them, install requirements_documents.txt and run python build_documents.py. PDFs were rendered and visually checked after generation. HTML versions contain the same report text and use relative image paths.'),
 h('2 File map'),
 p('Start with 00_START_HERE.html. The numbered folders contain the paper, experiment code, results, figures, research notes, and unchanged original project, respectively. MANIFEST_SHA256.txt records the packaged file hashes. The contents page links directly to the main documents, data, and reproduction entry points.'),
 h('3 Counts and independence'),
 p('The main study contains 5 environments x 20 seed values x 15 configurations = 1,500 runs. The planning ablation adds 5 x 20 x 3 retention rules x 2 planning settings = 600 runs. There are 6.3 million real training interactions and 21,000 evaluation checkpoints. The neural extension adds 20 x 7 = 140 model-training runs, 224,000 arriving examples, and 560 checkpoints. The complete package therefore contains 2,240 completed training runs.'),
 p('Seed values are paired across methods, and some environments deliberately share structure. The runs are separately initialized repetitions, not 2,240 independent problem definitions. Bootstrap analysis treats seed indices as paired blocks. No confidence interval pools the neural classifier and control-agent scores.')
])
supp.append([
 h('4 Algorithms and evaluation details'),
 sub('Transition model'),
 p('Sparse model keys contain (state, action, next state, terminal). Each key stores an observed count and reward sum. Dividing key counts by the total count for the state-action pair estimates the outcome distribution. Dividing reward sums by the same denominator estimates the contribution to expected immediate reward. Removing a buffer item subtracts exactly its contribution, and empty entries are deleted.'),
 sub('Coverage buffer pseudocode'),
 ('code','append new transition<br/>increment count for its state-action group<br/>if length exceeds capacity:<br/>&nbsp;&nbsp;find maximum group occupancy<br/>&nbsp;&nbsp;scan records from oldest to newest<br/>&nbsp;&nbsp;remove first record in a maximum-size group<br/>&nbsp;&nbsp;decrement its group and model counts'),
 p('The reservoir uses Algorithm R: the t-th arrival is inserted into a full M-item buffer with probability M/t, replacing a uniform slot. Unlike recent memory, old items can persist indefinitely. Each retained item has equal inclusion probability in the observed stream, but the stream itself is policy-dependent and may emphasize unhelpful parts of the environment.'),
 sub('Exact finite-horizon evaluator'),
 p('For a fixed greedy policy, define success probability at horizon zero as zero. At each subsequent horizon, sum the probability of immediate terminal success and the probability of each nonterminal outcome multiplied by its previous-horizon success probability. Expected undiscounted return is computed analogously, adding immediate reward before continuation. Repeat for 100 steps and read the start-state values. Only the evaluator accesses the true transition kernel. It never modifies the agent.'),
 p('This evaluator is checked against a closed-form three-step geometric-success example. Additional tests enforce buffer capacity, reconstruct model counts and rewards from retained records after every insertion, check terminal bootstrapping, verify seed repeatability, and ensure all test environments have a near-certain reachable successful policy.'),
 sub('Primary statistic'),
 p('For each run, trapezoidal integration over the ten registered checkpoints is divided by 3,000. An area of 1 would mean perfect 100-step success throughout training. The primary difference is coverage area minus recent-memory area at capacity 128. Differences are averaged over four stationary tasks within each seed block, and 20 blocks are resampled with replacement 10,000 times. Percentiles 2.5 and 97.5 form the interval. This is uncertainty over training randomness, conditional on the fixed suite.'),
 sub('Endpoint caution'),
 p('Success within 100 steps is forgiving once a policy can reach the goal. Expected return is retained as a secondary efficiency measure because unnecessary steps reduce it. The primary success-area statistic captures learning speed but does not fully describe path quality. A 100% success ceiling should not be interpreted as identical behavior.')
])
supp.append([
 h('5 Data dictionary and accounting'),
 table(['File and fields','Meaning'],[
 ['runs.csv identifiers','environment, method, capacity, seed, sweeps, every, family identify one training run. Capacity 0 means no raw buffer for Q learning or full model.'],
 ['step and phase','Completed real training interactions and environment phase. Step 1500 is still the old goal; step 1750 is the first measured new-goal checkpoint.'],
 ['success and return_value','Exact 100-step greedy success probability and expected undiscounted cumulative reward.'],
 ['train_seconds','Cumulative training time excluding evaluations and deep memory inspection.'],
 ['backups and plans','Cumulative scalar action-value updates and planning events. Backups are not hardware operations.'],
 ['agent_bytes','Raw deep Python object-size estimate; see the array-base correction below.'],
 ['retained and retained_pairs','Stored raw records and distinct retained state-action groups.'],
 ['model_edges and model_pairs','Distinct model outcomes and state-action pairs represented in sparse counts.'],
 ['run_metrics.csv','Per-run areas, final performance, corrected state bytes, work counts, and timing.'],
 ['summary_by_environment.csv','Means and 95% bootstrap intervals for all 105 environment/configuration combinations.'],
 ['paired_effects.csv','Exploratory within-environment coverage-minus-baseline contrasts.'],
 ['neural_runs.csv','Per-block test scores, class-specific accuracy, examples processed, and replay-buffer bytes.']
 ],[160,340]),
 p('The raw recursive counter visits ndarray objects but does not follow their base attribute. Model planning creates exactly one two-dimensional action-value view over a one-dimensional float64 allocation. On the execution runtime, analysis adds 112 + 8 x states x actions bytes after the first plan. Q-learning tables own their data and need no correction. Both the correction and final corrected estimate are explicit columns; raw source data are unchanged.'),
 p('Memory estimates include retained Python bookkeeping but not temporary allocations or full process RSS. The neural memory_bytes field measures the replay buffer only, not network parameters or transient gradients; it is intentionally not pooled with the control-agent state estimate. Timing confidence intervals describe variation among the observed runs and cannot compensate for hardware or implementation bias.'),
 h('6 Changes from the supplied project'),
 p('The expansion replaces episode budgets with exact interaction budgets, uses consistent planning settings within comparisons, records terminal outcomes explicitly, and evaluates policies exactly. It adds stronger replay and retention baselines, multiple environments, a changed objective, planning-budget ablations, uncertainty estimates, total-state accounting, and a separate neural experiment. The original project is preserved rather than silently rewritten. Its old figures are historical artifacts and are not used as evidence for the new paper.')
])
supp.append([
 h('7 Proposed local LLM validation'),
 p('This section is an unexecuted protocol for a follow-up study. No downloaded language model, model API, or LLM-generated action was used in the reported experiments. The finite-state tool workflow tests action prerequisites and retries only; it does not test language interpretation or hallucination.'),
 sub('Task and model controls'),
 p('Use two frozen locally hosted instruction models that fit the available hardware. Record checkpoint hashes, quantization, context limit, decoding parameters, and inference runtime. Build 30 held-out tool tasks in three families: structured-data transformation, file operations in an isolated workspace, and multi-step query workflows. Use a separate calibration set to fix prompts and memory rules. Include changed tool versions in a designated adaptation subset.'),
 p('Compare four conditions with identical models, action prompts, and tool contracts: no cross-task memory, recent memory, reservoir memory, and coverage-based memory. Within-episode action history must be handled identically in all conditions. Set the stored and retrieved memory limits in tokens, not record counts. Coverage groups should use only observable task features; their construction cost must be counted. Do not introduce an extra LLM summarizer to one condition without charging its cost.'),
 sub('Data and leakage controls'),
 p('Reset cross-task memory at the beginning of each independently seeded task sequence. All methods see the same task ordering. Store only feedback available to the acting agent; hidden evaluator solutions and held-out success labels must never become memory inputs. A reused task template must not reveal the exact held-out answer. Version-tag each tool outcome so obsolete records can be recognized. Record all failed calls, retries, parsing failures, and timeouts.'),
 sub('Metrics and prospective decision rule'),
 p('For each model, compare success with a paired task bootstrap and report token usage, tool calls, invalid actions, latency, memory size, and retrieval cost. The proposed efficiency criterion is a lower 95% confidence bound on success difference above -3 percentage points, together with at least 20% lower measured total inference tokens than a recent-memory baseline. These are design targets for a new study, not achieved results. A 30-task test may be underpowered; report an inconclusive outcome rather than treating non-significance as equivalence.'),
 p('Start with a free local feasibility pilot to measure runtime and task difficulty. The final experiment would require new held-out tasks after any pilot-driven tuning. If the hardware cannot execute a suitable model, keep this application as proposed work instead of substituting scripted behavior and calling it an LLM.'),
 h('8 Research claims that remain open'),
 p('Open questions include retention under noisy state abstraction, partial observability, adaptive memory budgets, hybrid recent and diverse archives, and stronger compute-matched baselines. Evidence from this package does not establish parameter compression, lower pretraining GPU cost, energy savings, safety of autonomous external actions, production reliability, or superiority to current foundation models. Those claims require different measurements and experiments.')
])

def footer(canvas,doc):
    canvas.saveState();canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#66717D'))
    canvas.drawString(54,28,COMPANY+'  |  Private research  |  Authorized recipients only')
    canvas.drawRightString(558,28,str(doc.page-1));canvas.restoreState()

def cover(canvas,doc,title,kind):
    canvas.saveState()
    canvas.setFillColor(colors.white);canvas.rect(0,0,612,792,fill=1,stroke=0)
    canvas.drawImage(str(OUT/'assets/company_logo.png'),186,442,width=240,height=300,mask='auto')
    def centered(text,top,size=12,leading=16,font='Helvetica',color='#202B35',width=480):
        sty=ParagraphStyle('CoverLine',fontName=font,fontSize=size,leading=leading,alignment=TA_CENTER,textColor=colors.HexColor(color))
        para=Paragraph(html.escape(text),sty)
        _,height=para.wrap(width,300)
        para.drawOn(canvas,(612-width)/2,top-height)
        return top-height
    centered('RESEARCH BY '+COMPANY.upper(),443,13,18,'Helvetica-Bold')
    centered(title,393,25,29,'Times-Bold')
    centered(kind+'',282,10,15)
    centered(BRAND['owner_line'],253,11,16)
    centered('PRIVATE RESEARCH',186,12,17,'Helvetica-Bold')
    centered(BRAND['notice'],156,10,15,width=430)
    canvas.restoreState()

def build(stem,pages):
    story=[Spacer(1,1),PageBreak()]
    document_kind='Research report' if stem=='Research_Paper' else 'Technical supplement'
    html_pages=['<section class="cover"><img src="assets/company_logo.png" alt="Company logo"><p class="brand">Research by '+html.escape(COMPANY)+'</p><h1>'+html.escape(pages[0][0][1])+'</h1><p>'+document_kind+'</p><p>'+html.escape(BRAND['owner_line'])+'</p><h2>Private research</h2><p class="notice">'+html.escape(BRAND['notice'])+'</p></section>']
    for pi,page in enumerate(pages):
        if pi:story.append(PageBreak())
        hb=[]
        for b in page:
            kind=b[0]
            if kind in ['title','subtitle','p','h','sub','ref','code']:
                style={'title':'PaperTitle','subtitle':'PaperSub','p':'BodyPaper','h':'HeadPaper','sub':'SubPaper','ref':'RefPaper','code':'CodePaper'}[kind]
                story.append(Paragraph(b[1],styles[style]))
                tag={'title':'h1','subtitle':'p','p':'p','h':'h2','sub':'h3','ref':'p','code':'pre'}[kind]
                text=b[1].replace('<link href=', '<a href=').replace('</link>','</a>')
                hb.append(f'<{tag}>{text}</{tag}>')
            elif kind=='table':
                _,headers,rows,widths=b
                cells=[[Paragraph(html.escape(str(x)),styles['CellPaper']) for x in r] for r in [headers,*rows]]
                t=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E5EDF1')),('GRID',(0,0),(-1,-1),.5,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
                story.extend([t,Spacer(1,10)])
                hb.append('<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>')
            elif kind=='fig':
                _,filename,caption,width=b
                path=ROOT/'04_Figures'/filename
                w,hp=ImageReader(str(path)).getSize()
                story.append(KeepTogether([Image(str(path),width=width,height=hp*width/w),Paragraph(caption,styles['CaptionPaper'])]))
                hb.append(f'<figure><img src="../04_Figures/{filename}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>')
        html_pages.append('<section>'+''.join(hb)+'</section>')
    doc=SimpleDocTemplate(str(OUT/(stem+'.pdf')),pagesize=(612,792),rightMargin=54,leftMargin=54,topMargin=44,bottomMargin=48,title=pages[0][0][1],author=COMPANY,subject='Private research - authorized recipients only')
    doc.build(story,onFirstPage=lambda c,d:cover(c,d,pages[0][0][1],document_kind),onLaterPages=footer)
    css='body{max-width:850px;margin:35px auto;padding:0 24px;color:#151a20;font:18px/1.55 Georgia,serif}h1,h2,h3{color:#000}h1{font-size:34px;line-height:1.15}h2{font-size:24px;margin-top:28px}h3{font-size:19px}table{border-collapse:collapse;width:100%;font:14px/1.45 Arial,sans-serif;margin:18px 0}td,th{border:1px solid #d9d9d9;padding:9px;text-align:left}th{background:#e5edf1}figure{margin:20px 0}img{max-width:100%}figcaption{font:13px/1.5 Arial,sans-serif;color:#424b56}a{color:#145a70}pre{white-space:pre-wrap;font:13px/1.5 monospace}section{margin-bottom:40px}@media print{body{font-size:11pt;max-width:none;margin:0}section{break-before:page}section:first-child{break-before:auto}}'
    css+=' .cover{text-align:center;padding:20px 0 60px}.cover img{width:240px;height:300px;object-fit:contain}.cover .brand{font: bold 15px Arial,sans-serif;text-transform:uppercase;letter-spacing:1px}.cover h1{max-width:620px;margin:35px auto}.cover h2{font: bold 15px Arial,sans-serif;text-transform:uppercase;margin-top:55px}.cover .notice{max-width:580px;margin:15px auto;font:15px/1.6 Arial,sans-serif}'
    (OUT/(stem+'.html')).write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(pages[0][0][1])+'</title><style>'+css+'</style></head><body>'+''.join(html_pages)+'</body></html>',encoding='utf-8')
    print(stem, 'planned pages',len(pages)+1)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    build('Research_Paper',paper)
    build('Technical_Supplement',supp)
