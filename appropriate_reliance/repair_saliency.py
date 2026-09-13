"""Recompute saliency for the displayed class on frozen source units, without LLMs."""
import ast,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer
FEATURES=['age','workclass','education','marital-status','occupation','native-country','hours-per-week','gender','race']
CATS=[c for c in FEATURES if c not in ['age','hours-per-week']]
MAP={19:'physician',21:'professor',22:'psychologist',25:'surgeon',26:'teacher'}

def main():
 out=ROOT/'appropriate_reliance/data_v2';out.mkdir(exist_ok=True)
 source=pd.read_csv(ROOT/'appropriate_reliance/datasets/adult.csv')[FEATURES+['income']].copy()
 for c in source:
  if source[c].dtype==object or str(source[c].dtype)=='str':source[c]=source[c].astype(str).str.strip()
 y=(source.income=='>50K').astype(int).to_numpy(); raw=source[FEATURES].copy();encoded=raw.copy();encoders={}
 for c in CATS:
  encoders[c]=LabelEncoder().fit(raw[c]);encoded[c]=encoders[c].transform(raw[c])
 raw_bios=pd.read_csv(ROOT/'appropriate_reliance/datasets/train_df.csv').dropna()
 raw_bios=raw_bios[raw_bios.profession.isin(MAP)].copy(); labels=raw_bios.profession.map(MAP)
 label_encoder=LabelEncoder().fit(labels)
 counts={};hashes={}
 for seed in range(26):
  print('seed',seed,flush=True)
  if (out/f'adult_neighbors_and_explanations_seed_{seed}.csv').exists() and (out/f'bios_neighbors_and_explanations_seed_{seed}.csv').exists():
   continue
  a=pd.read_csv(ROOT/f'appropriate_reliance/data/adult_neighbors_and_explanations_seed_{seed}.csv')
  train,test=train_test_split(np.arange(len(raw)),train_size=.8,stratify=y,random_state=seed)
  x=encoded.to_numpy()
  # Preserve the legacy encoder vocabulary fitted to the full source X.
  ct=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),[FEATURES.index(c) for c in CATS])],remainder='passthrough')
  xt=ct.fit(x).transform(x[train]);model=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=seed,n_jobs=1).fit(xt,y[train])
  for idx,row in a.iterrows():
   hr=ast.literal_eval(row.test_instance);v=np.array([encoders[c].transform([hr[c]])[0] if c in CATS else hr[c] for c in FEATURES],dtype=float)
   prediction=int(model.predict(ct.transform(v.reshape(1,-1)))[0])
   a.at[idx,'test_predicted_label']=prediction
   for neighbor in [1,2]:
    nh=ast.literal_eval(row[f'neighbor{neighbor}_instance']);nv=np.array([encoders[c].transform([nh[c]])[0] if c in CATS else nh[c] for c in FEATURES],dtype=float)
    a.at[idx,f'neighbor{neighbor}_predicted_label']=int(model.predict(ct.transform(nv.reshape(1,-1)))[0])
   stable_seed=int(hashlib.sha256(f'adult:{seed}:{idx}:saliency-v2'.encode()).hexdigest()[:8],16)
   exp=LimeTabularExplainer(x[train],feature_names=FEATURES,class_names=['<=50K','>50K'],
     categorical_features=[FEATURES.index(c) for c in CATS],categorical_names={FEATURES.index(c):list(encoders[c].classes_) for c in CATS},kernel_width=3,random_state=stable_seed)
   explanation=exp.explain_instance(v,lambda batch:model.predict_proba(ct.transform(batch)),labels=(prediction,),num_features=6,num_samples=5000)
   a.at[idx,'lime_explanation']=repr([(str(k),float(v)) for k,v in explanation.as_list(label=prediction)])
   a.at[idx,'explanation_target_label']=prediction;a.at[idx,'explanation_seed']=stable_seed
  a.to_csv(out/f'adult_neighbors_and_explanations_seed_{seed}.csv',index=False)
  b=pd.read_csv(ROOT/f'appropriate_reliance/data/bios_neighbors_and_explanations_seed_{seed}.csv')
  vectorizer=CountVectorizer(stop_words='english',max_features=5000)
  xb=vectorizer.fit_transform(raw_bios.hard_text)
  clf=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=seed,n_jobs=1).fit(xb,label_encoder.transform(labels))
  pipeline=make_pipeline(vectorizer,clf)
  for idx,row in b.iterrows():
   prediction=int(pipeline.predict([row.test_instance])[0]);label=str(label_encoder.inverse_transform([prediction])[0])
   b.at[idx,'test_predicted_label']=label
   for neighbor in [1,2]:
    b.at[idx,f'neighbor{neighbor}_predicted_label']=str(label_encoder.inverse_transform(pipeline.predict([row[f'neighbor{neighbor}_instance']]))[0])
   stable_seed=int(hashlib.sha256(f'bios:{seed}:{idx}:saliency-v2'.encode()).hexdigest()[:8],16)
   exp=LimeTextExplainer(class_names=label_encoder.classes_.tolist(),random_state=stable_seed)
   explanation=exp.explain_instance(row.test_instance,pipeline.predict_proba,labels=(prediction,),num_features=6,num_samples=5000)
   b.at[idx,'lime_explanation']=repr([(str(k),float(v)) for k,v in explanation.as_list(label=prediction)])
   b.at[idx,'explanation_target_label']=label;b.at[idx,'explanation_seed']=stable_seed
  b.to_csv(out/f'bios_neighbors_and_explanations_seed_{seed}.csv',index=False)
 for p in sorted(out.glob('*.csv')):hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 (out/'manifest.json').write_text(json.dumps({'version':'reliance-v2-target-class-saliency','lime_version':'0.2.0.1','num_samples':5000,'files':hashes,'class_order':label_encoder.classes_.tolist(),'predictions_reconstructed':832},indent=2))
if __name__=='__main__':main()
