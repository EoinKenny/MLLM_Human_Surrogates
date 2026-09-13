"""Versioned, model-verified German Credit recourses. No LLM calls.

Codebook: https://archive.ics.uci.edu/dataset/144/statloggermancreditdata
The historical Statlog codebook is retained explicitly, not silently replaced
by the distinct corrected South German dataset. Numeric categories are never
interpreted as ordinal improvements. Every plan must flip the SAME classifier.
"""
from pathlib import Path
import argparse, hashlib, itertools, json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

VERSION = 'recourse-v2-codebook-model-verified'
COLUMNS = ['status','duration','credit_history','purpose','credit_amount','savings',
'employment','installment_rate','personal_status','guarantors','residence_since',
'property','age','other_plans','housing','existing_credits','job','liable_people',
'telephone','foreign_worker','credit_risk']
CODEBOOK = {
'status': dict(zip(['A11','A12','A13','A14'], ['Below 0 DM','0 to less than 200 DM','At least 200 DM, or salary assigned to this account for at least one year','No checking account'])),
'credit_history': dict(zip(['A30','A31','A32','A33','A34'], ['No previous credit, or all credit repaid on time','All credit at this bank repaid on time','Existing credit repaid on time so far','Past delays in repayment','Critical account, or other credit outside this bank'])),
'purpose': dict(zip(['A40','A41','A42','A43','A44','A45','A46','A47','A48','A49','A410'], ['New car','Used car','Furniture or equipment','Radio or television','Domestic appliances','Repairs','Education','Vacation (historical codebook category)','Retraining','Business','Other purpose'])),
'savings': dict(zip(['A61','A62','A63','A64','A65'], ['Less than 100 DM','100 to less than 500 DM','500 to less than 1,000 DM','At least 1,000 DM','Unknown, or no savings account'])),
'employment': dict(zip(['A71','A72','A73','A74','A75'], ['Unemployed','Less than one year with current employer','One to less than four years with current employer','Four to less than seven years with current employer','At least seven years with current employer'])),
'personal_status': dict(zip(['A91','A92','A93','A94','A95'], ['Male, divorced or separated','Female, divorced, separated or married','Male, single','Male, married or widowed','Female, single'])),
'guarantors': {'A101':'No co-applicant or guarantor','A102':'Co-applicant','A103':'Guarantor'},
'property': {'A121':'Real estate','A122':'Building-society savings agreement or life insurance; no real estate','A123':'Car or other property; neither of the preceding categories','A124':'Unknown, or no property'},
'other_plans': {'A141':'At a bank','A142':'At stores','A143':'None'},
'housing': {'A151':'Rented accommodation','A152':'Own home','A153':'Accommodation provided without rent'},
'job': {'A171':'Unemployed or unskilled non-resident','A172':'Unskilled resident','A173':'Skilled employee or official','A174':'Management, self-employed, or highly qualified employee'},
'telephone': {'A191':'None','A192':'Registered in the applicant’s name'},
'foreign_worker': {'A201':'Yes','A202':'No'},
}
DISPLAY = dict(zip(COLUMNS, ['Checking-account balance','Loan duration','Credit repayment history','Loan purpose','Requested loan amount','Savings account or bonds','Time with current employer','Monthly installment as a share of disposable income','Personal status (historical dataset category)','Other debtor or guarantor','Time at current residence','Property','Age','Other installment plans','Housing','Existing loans at this bank','Job category','People financially supported','Telephone','Foreign-worker status','Credit risk']))
# Small directed edits; no assumptions that unknown savings or no account is worse.
# Credit history, age, employment tenure and housing are immutable. Increasing
# tenure while freezing age or buying a home while freezing property is inconsistent.
TRANSITIONS = {
 'status': {'A11':['A12'], 'A12':['A13']},
 'savings': {'A61':['A62'], 'A62':['A63'], 'A63':['A64']},
}
NUMERIC = ['duration','credit_amount','existing_credits']
MUTABLE = set(TRANSITIONS) | set(NUMERIC)

def canonical(x): return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def digest(x): return hashlib.sha256(canonical(x).encode()).hexdigest()

def load_data(path):
    df = pd.read_csv(path,sep=r'\s+',header=None,names=COLUMNS)
    if len(df.columns)!=21 or df.empty or df.isna().any().any():
        raise ValueError('Incomplete German Credit source')
    for col, mapping in CODEBOOK.items():
        unknown=set(df[col])-set(mapping)
        if unknown: raise ValueError(f'Unknown categories in {col}: {sorted(unknown)}')
    for c in set(COLUMNS)-set(CODEBOOK):
        df[c]=pd.to_numeric(df[c],errors='raise')
        if not np.isfinite(df[c]).all(): raise ValueError(f'Nonfinite {c}')
    if set(df.credit_risk)!={1,2}: raise ValueError('Expected 1=good, 2=bad credit risk')
    if (df[NUMERIC]<=0).any().any(): raise ValueError('Invalid positive numeric source')
    return df

def distance(a,b,stds):
    changed=[c for c in a if a[c]!=b[c]]
    # L0 counts edited original fields, not one-hot columns. Mixed L1 uses one
    # unit per changed categorical field plus training-SD-scaled numeric edits.
    l1=sum(1.0 if c in CODEBOOK else abs(float(a[c])-float(b[c]))/stds[c] for c in changed)
    return len(changed),float(l1)

def validate_transition(a,b,bounds):
    if set(a)!=set(COLUMNS[:-1]) or set(b)!=set(a): raise ValueError('Missing profile fields')
    for c in a:
        if c in CODEBOOK:
            if a[c] not in CODEBOOK[c] or b[c] not in CODEBOOK[c]: raise ValueError(f'Unknown {c}')
        else:
            if not np.isfinite(b[c]) or int(b[c])!=b[c]: raise ValueError(f'Invalid numeric {c}')
        if a[c]==b[c]: continue
        if c not in MUTABLE: raise ValueError(f'Immutable attribute changed: {c}')
        if c in TRANSITIONS and b[c] not in TRANSITIONS[c].get(a[c],[]): raise ValueError(f'Invalid {c} transition')
        if c in NUMERIC and not bounds[c][0]<=b[c]<a[c]: raise ValueError(f'Invalid reduction of {c}')
    if a==b: raise ValueError('No change cannot be decision-changing recourse')

def readable(col,value):
    if col in CODEBOOK: return CODEBOOK[col][value]
    v=f'{int(value):,}'
    return v+ {'duration':' months','credit_amount':' DM','age':' years',
       'residence_since':' year(s)','installment_rate':'%','existing_credits':' loan(s)',
       'liable_people':' person(s)'}.get(col,'')

def create_readable_prompt(original, recourse_instance, d_name='german'):
    profile='\n'.join(f'- {DISPLAY[c]}: {readable(c,original[c])}' for c in COLUMNS[:-1])
    changes='\n'.join(f'{i}. {DISPLAY[c]}: change from {readable(c,original[c])} to {readable(c,recourse_instance[c])}.'
      for i,c in enumerate([c for c in original if original[c]!=recourse_instance[c]],1))
    return f'''Imagine you are the loan applicant described below. This is a historical German Credit scenario; monetary values use Deutsche Mark (DM), not today's currency or prices.
Use only the supplied information. The profile describes the loan you requested, including its actual purpose, amount and duration.

YOUR CURRENT PROFILE
{profile}

The credit classifier rejected this profile. The following changes would make that same classifier accept the revised profile. Everything not listed stays as shown above. Judge whether you would realistically undertake the plan; classifier acceptance alone does not establish affordability or guarantee real-world approval.

PROPOSED PLAN
{changes}

Rate each question from 1 to 7: 1 = strongly no, 4 = neutral, 7 = strongly yes.
1. Is the AI system's plan a reasonable explanation for the rejection of your loan application?
2. Would you carry out the plan to obtain loan approval?
Explain your reasoning and return a JSON object with exactly these fields:
{{"reasoning": "your explanation", "acceptance_score": 1, "actionability_score": 1}}
Replace the example scores with your own integer ratings from 1 to 7.'''

def select_uniform_recourses(records,n_final=200,min_per_l0=0,seed=42):
    unique={r['instance_id']:r for r in records}
    if len(unique)<n_final: raise ValueError(f'Need {n_final} unique recourses; only {len(unique)} exist')
    # Round robin across L0 strata, then occupied L1 bins. Fine bins near zero
    # deliberately allocate more coverage there. Remainders prefer low distance.
    edges=[0,.05,.1,.2,.3,.5,.75,1,1.5,2,3,4,5,6,8,10,13,float('inf')]
    cells={}
    for r in sorted(unique.values(),key=lambda r:r['instance_id']):
        b=int(np.searchsorted(edges,r['proximity'],side='right')-1)
        cells.setdefault(r['sparsity'],{}).setdefault(b,[]).append(r)
    rng=np.random.default_rng(seed)
    for bins in cells.values():
        for bucket in bins.values(): rng.shuffle(bucket)
    counts={k:0 for k in cells};used_bins={k:{} for k in cells}; selected=[]; seen_prompts=set()
    while len(selected)<n_final:
        levels=[k for k,v in cells.items() if any(v.values())]
        if not levels: raise ValueError('Insufficient unique rendered prompts')
        k=min(levels,key=lambda k:(counts[k],k))
        bins=cells[k];b=min((b for b in bins if bins[b]),key=lambda b:(used_bins[k].get(b,0),b))
        r=bins[b].pop()
        prompt_hash=digest(create_readable_prompt(r['original_instance'],r['recourse_instance']))
        if prompt_hash in seen_prompts: continue
        seen_prompts.add(prompt_hash);selected.append(r);counts[k]+=1;used_bins[k][b]=used_bins[k].get(b,0)+1
    return selected

def select_spanning_recourses(records,n_final=2000):
    return select_uniform_recourses(records,min(n_final,len({r['instance_id'] for r in records})))

def generate(path,n=200,seed=42):
    df=load_data(path);X=df.drop(columns='credit_risk');y=(df.credit_risk==1).astype(int)
    train,test=train_test_split(np.arange(len(df)),test_size=.3,random_state=seed,stratify=y)
    cat=list(CODEBOOK);num=[c for c in X if c not in cat]
    transformer=ColumnTransformer([('cat',OneHotEncoder(categories=[list(CODEBOOK[c]) for c in cat],handle_unknown='error',sparse_output=False),cat),('num','passthrough',num)])
    model=make_pipeline(transformer,RandomForestClassifier(n_estimators=100,max_depth=12,random_state=seed,n_jobs=1))
    model.fit(X.iloc[train],y.iloc[train])
    stds={c:float(X.iloc[train][c].std()) for c in num}
    if any(v<=0 for v in stds.values()): raise ValueError('Zero numeric training variance')
    bounds={c:(max(1,int(X.iloc[train][c].min())),int(X.iloc[train][c].max())) for c in NUMERIC}
    predictions=model.predict(X.iloc[test]);rejected=sorted(test[predictions==0].tolist())
    records={}
    targets=[.01,.025,.05,.1,.2,.3,.5,.75,1,1.5,2,3,4,6,8,10,13]
    for row_id in rejected:
        a=X.iloc[row_id].to_dict();opts={c:TRANSITIONS[c].get(a[c],[]) for c in TRANSITIONS}
        for c in NUMERIC:
            values=[int(a[c])-max(1,int(round(t*stds[c]))) for t in targets]
            opts[c]=sorted({v for v in values if bounds[c][0]<=v<a[c]},reverse=True)
        active=[c for c in opts if opts[c]]
        candidates={}
        for size in range(1,len(active)+1):
            for subset in itertools.combinations(active,size):
                for t in targets:
                    b=a.copy()
                    for c in subset:
                        if c in TRANSITIONS: b[c]=opts[c][0]
                        else:
                            goal=int(a[c])-max(1,int(round(t*stds[c]/max(1,sum(z in NUMERIC for z in subset)))))
                            b[c]=min(opts[c],key=lambda v:(abs(v-goal),-v))
                    l0,l1=distance(a,b,stds)
                    if l1<=13: candidates[canonical(b)]=b
        if not candidates: continue
        bs=list(candidates.values()); ps=model.predict(pd.DataFrame(bs,columns=X.columns))
        for b,pred in zip(bs,ps):
            if pred!=1: continue
            validate_transition(a,b,bounds);l0,l1=distance(a,b,stds)
            identity=digest({'version':VERSION,'source_row':row_id,'original':a,'proposed':b})
            records[identity]={'instance_id':identity,'user_id':row_id,'source_row':row_id,'original_instance':a,
                'recourse_instance':b,'sparsity':l0,'proximity':l1,'original_prediction':0,'recourse_prediction':1,
                'classifier_id':f'random-forest-seed-{seed}','prompt_version':VERSION}
    selected=select_uniform_recourses(list(records.values()),n,seed=seed)
    # Independently re-evaluate every exported plan, including original rejection.
    originals=pd.DataFrame([r['original_instance'] for r in selected],columns=X.columns)
    proposals=pd.DataFrame([r['recourse_instance'] for r in selected],columns=X.columns)
    if not (model.predict(originals)==0).all() or not (model.predict(proposals)==1).all(): raise ValueError('Decision flip failed')
    for r in selected:r['prompt_text']=create_readable_prompt(r['original_instance'],r['recourse_instance']);r['action_id']=r['instance_id']
    manifest={'version':VERSION,'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'seed':seed,
      'model':model[-1].get_params(),'train_rows':train.tolist(),'test_rows':test.tolist(), 'positive_class':'1=good credit risk',
      'training_stds':stds,'numeric_bounds':bounds,'distance':'L0: changed source features. Mixed L1: numeric differences / training SD, plus 1 per categorical change.',
      'selection':'equal L0 strata, round robin occupied L1 bins, finer bins near zero; without replacement',
      'candidate_count':len(records),'rejected_test_rows':len(rejected),'selected_count':len(selected),
      'l0_counts':{str(k):sum(r['sparsity']==k for r in selected) for k in sorted({r['sparsity'] for r in selected})},
      'l1_min':min(r['proximity'] for r in selected),'l1_max':max(r['proximity'] for r in selected)}
    return selected,manifest,model

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--n',type=int,default=200);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows,manifest,model=generate(Path(__file__).parent/'datasets/german.data',args.n)
    (args.output/'recourses.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    pd.DataFrame(rows).to_csv(args.output/'german_experiment_prompts_actionable.csv',index=False)
    import joblib
    joblib.dump(model,args.output/'classifier.joblib')
    print(json.dumps(manifest,indent=2))
