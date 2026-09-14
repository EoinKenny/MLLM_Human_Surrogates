"""Reproduce paired n=30 bootstrap plots from the final long-format CSV."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
OUT=args.output; OUT.mkdir(parents=True,exist_ok=True)
with args.data.open(newline='') as f: rows=list(csv.DictReader(f))
models=list(dict.fromkeys(r['model'] for r in rows));scores={};invalid={};strict={}
conds=['control','causal','cfs']; families={r['model']:r['model_family'] for r in rows}
reference=None
for m in models:
    ds=[r for r in rows if r['model']==m]
    by={(int(r['question_index']),r['condition']):r for r in ds}
    assert len(ds)==len(by)==900
    keys=[(by[i,'control']['question_id'],by[i,'control']['ground_truth']) for i in range(1,301)]
    if reference is None:reference=keys
    assert keys==reference
    a=np.array([[int(by[i,c]['correct']) for c in conds] for i in range(1,301)])
    scores[m]=a; invalid[m]=[sum(1-int(by[i,c]['answer_valid']) for i in range(1,301)) for c in conds]
    strict[m]=[sum(str(json.loads(by[i,c]['saved_label_json'])) in ('0','1') and str(json.loads(by[i,c]['saved_label_json']))==by[i,c]['ground_truth'] for i in range(1,301))/300 for c in conds]
assert len(models)==12
rng=np.random.default_rng(42);indices=rng.integers(0,300,size=(1000,30));np.save(OUT/'bootstrap_indices.npy',indices)
boots={m:scores[m][indices].mean(axis=1)*100 for m in models}
summary={}
for m in models:
 b=boots[m];summary[m]={'accuracy_percent':(scores[m].mean(0)*100).tolist(),'invalid':invalid[m],'strict_accuracy_percent':(np.array(strict[m])*100).tolist(),'bootstrap_mean_percent':b.mean(0).tolist(),'bootstrap_central95_percent':np.percentile(b,[2.5,97.5],axis=0).tolist(),'cf_minus_control':{'mean_pp':float((b[:,2]-b[:,0]).mean()),'central95_pp':np.percentile(b[:,2]-b[:,0],[2.5,97.5]).tolist(),'fraction_positive':float(np.mean(b[:,2]>b[:,0]))}}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
colors=['#6484a5','#df9655','#459b80'];fig,axes=plt.subplots(4,3,figsize=(12,12),sharey=True)
for ax,m in zip(axes.flat,models):
 b=boots[m];bp=ax.boxplot([b[:,j] for j in range(3)],positions=[1,2,3],widths=.48,patch_artist=True,whis=(2.5,97.5),showfliers=False,medianprops={'color':'black','linewidth':1.3})
 for patch,color in zip(bp['boxes'],colors):patch.set_facecolor(color);patch.set_alpha(.65)
 ax.scatter([1,2,3],scores[m].mean(0)*100,color='black',s=24,zorder=5)
 ax.set_title(m+(' · API' if families[m]=='API' else ' · open weights'),fontsize=10,fontweight='bold');ax.set_xticks([1,2,3],['Control','Causal','CF']);ax.set_ylim(15,113);ax.set_yticks([20,40,60,80,100]);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
 for j,val in enumerate(scores[m].mean(0)*100):ax.text(j+1,107,f'{val:.1f}%',ha='center',fontsize=9)
fig.suptitle('Teaching accuracy with samples of 30 questions',fontsize=17,fontweight='bold',y=.985)
fig.text(.5,.952,'1,000 paired bootstrap resamples from the same 300 questions · seed 42',ha='center',fontsize=11)
fig.supylabel('Accuracy (%)',x=.015)
fig.text(.5,.02,'Boxes: middle 50% · whiskers: middle 95% · dots and labels: observed accuracy over all 300 questions\nDecimal labels accepted; invalid answers count as incorrect. These distributions describe n=30 samples, not confidence intervals for n=300 accuracy.',ha='center',fontsize=9)
fig.tight_layout(rect=[.03,.065,1,.94]);fig.savefig(OUT/'bootstrap_n30_1000.png',dpi=180);fig.savefig(OUT/'bootstrap_n30_1000.pdf');plt.close(fig)
lines=['# Twelve-model teaching comparison','', 'Same 300 questions and ground truths verified across all 12 models. Conditions are ordered control, causal, counterfactual. API condition names are taken from the export; the export does not include full prompts to independently verify wording.','', '| Model | Control | Causal | Counterfactual | CF − control |','|---|---:|---:|---:|---:|']
for m in models:
 a=summary[m]['accuracy_percent'];lines.append(f'| {m} | {a[0]:.1f}% | {a[1]:.1f}% | {a[2]:.1f}% | {a[2]-a[0]:+.1f} pp |')
lines+=['','## Method','', 'Draw 30 question indices with replacement from 300, repeated 1,000 times with NumPy default_rng seed 42. Use identical indices across conditions and models to preserve pairing. No new inference. Score saved labels consistently: accept 0/1 and 0.0/1.0; invalid answers count as incorrect. GPT-4o has 12 invalid control and 6 invalid counterfactual answers; all remaining normalized labels are valid. Strict open-source parsing results are retained separately in summary.json.','', 'Boxplots show median and interquartile range, with 2.5th–97.5th percentile whiskers. Dots show observed full-sample accuracy. These are n=30 resampling distributions, not confidence intervals for full-sample accuracy and not independent experimental replications. No significance tests are inferred from bootstrap win fractions.']
(OUT/'results.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:18]));print('invalid',invalid)
