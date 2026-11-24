"""Future CPU baseline runner; preparation does not execute this model training."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from care_contract import LABELS,read_jsonl,write_jsonl,write_json,hash_inputs,dataset_errors,leakage_audit


def text(row, view):
    value=row['inputs']
    if view=='utterance_only':return value['user_input']
    if view=='history_without_state':return json.dumps({'history':value['history'],'user_input':value['user_input']},ensure_ascii=False)
    return json.dumps(value,ensure_ascii=False,sort_keys=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',required=True);p.add_argument('--evaluate',required=True);p.add_argument('--output',required=True)
    p.add_argument('--view',choices=['full_context','history_without_state','utterance_only'],default='full_context')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--regularization-c',type=float,default=1.)
    a=p.parse_args();train=read_jsonl(a.train);test=read_jsonl(a.evaluate)
    errors=dataset_errors(train+test);leaks=leakage_audit(train+test)
    if errors or leaks:raise ValueError('data validation or group leakage failure')
    if any(r['partition']!='train_gold' or r['annotation']['review_status']!='human_approved' for r in train):
        raise ValueError('gold-only baseline requires approved train_gold')
    if any(r['annotation']['review_status']!='human_approved' for r in test):raise ValueError('human gold evaluation required')
    train=[r for r in train if r['annotation']['primary_intent'] is not None]
    if {r['annotation']['primary_intent'] for r in train}!=set(LABELS):raise ValueError('all four classes required in train fold')
    model=Pipeline([('features',TfidfVectorizer(analyzer='char',ngram_range=(2,4),min_df=1,max_features=100000)),
                    ('classifier',LogisticRegression(C=a.regularization_c,max_iter=2000,random_state=a.seed))])
    model.fit([text(r,a.view) for r in train],[r['annotation']['primary_intent'] for r in train])
    fingerprint=hashlib.sha256(model['classifier'].coef_.tobytes()+model['classifier'].intercept_.tobytes()+
        json.dumps(model['features'].vocabulary_,sort_keys=True,default=int).encode()).hexdigest()
    values=model.predict_proba([text(r,a.view) for r in test]);classes=list(model.classes_);rows=[]
    for r,p in zip(test,values):
        probabilities={label:float(p[classes.index(label)]) for label in LABELS}
        rows.append({'record_id':r['record_id'],'input_hash':hash_inputs(r['inputs']),
                     'prediction':max(probabilities,key=probabilities.get),'probabilities':probabilities,
                     'format_valid':True,'action':None,'model_id':'char_tfidf_logistic_regression',
                     'model_seed':a.seed,'model_run_id':fingerprint,'input_view':a.view})
    write_jsonl(a.output,rows)
    write_json(str(a.output)+'.manifest.json',{'arguments':vars(a),'train_n':len(train),'test_n':len(test),
        'note':'CPU baseline. Identical full data may yield deterministic fits; repeated seeds are not independent stochastic evidence.'})


if __name__=='__main__':main()
