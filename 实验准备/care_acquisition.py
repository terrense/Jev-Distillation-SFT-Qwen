"""Training-pool-only candidate selection. This is active annotation, not OPD."""
import hashlib
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from care_contract import class_vector, assert_teacher_access


def select(rows, budget, method='hybrid', student_predictions=None, seed=42):
    if budget<1 or budget>len(rows):raise ValueError('budget must fit eligible pool')
    for row in rows:
        if row['partition']!='train_unlabeled':raise ValueError('selection cannot inspect dev/test/unassigned data')
        # Local selection does not itself upload records; external API access has a separate gate.
    if len({r['record_id'] for r in rows})!=len(rows):raise ValueError('duplicate acquisition candidates')
    rng=np.random.default_rng(seed)
    if method=='random':
        chosen=rng.choice(len(rows),size=budget,replace=False).tolist()
        return [{'record_id':rows[i]['record_id'],'method':method,'rank':rank+1,'requires_human_review':True} for rank,i in enumerate(chosen)]
    predictions=student_predictions or {};uncertainty=[];disagreement=[]
    for row in rows:
        prediction=predictions.get(row['record_id'])
        if not prediction or prediction.get('error'):raise ValueError('valid student probabilities required for non-random selection')
        p=sorted(class_vector(prediction.get('probabilities')),reverse=True)
        uncertainty.append(1-(p[0]-p[1]))
        teachers={x['prediction'] for x in row['teacher_outputs'] if x.get('prediction')}
        disagreement.append(float(len(teachers)>1 or bool(teachers) and prediction.get('prediction') not in teachers))
    base=np.asarray(uncertainty) if method=='margin' else .6*np.asarray(uncertainty)+.25*np.asarray(disagreement)
    if method not in {'margin','hybrid'}:raise ValueError('unknown selection method')
    # Context-aware lexical diversity, not a claim of semantic embedding equivalence.
    texts=[json.dumps(r['inputs'],ensure_ascii=False,sort_keys=True) for r in rows]
    vectors=TfidfVectorizer(analyzer='char',ngram_range=(2,4),min_df=1,max_features=30000).fit_transform(texts)
    available=np.ones(len(rows),dtype=bool);nearest=np.zeros(len(rows));chosen=[]
    # Ties are broken by a stable id hash, never the proposed/correct label.
    tie=np.array([int(hashlib.sha256((str(seed)+r['record_id']).encode()).hexdigest()[:8],16)/2**32 for r in rows])
    while len(chosen)<budget:
        scores=base+(.15*(1-nearest) if method=='hybrid' else 0)+tie*1e-10
        scores[~available]=-np.inf;i=int(np.argmax(scores));available[i]=False;chosen.append(i)
        similarity=(vectors@vectors[i].T).toarray().ravel();nearest=np.maximum(nearest,similarity)
    return [{'record_id':rows[i]['record_id'],'method':method,'rank':rank+1,
             'margin_uncertainty':float(uncertainty[i]),'teacher_student_disagreement':bool(disagreement[i]),
             'requires_human_review':True,'is_on_policy_distillation':False} for rank,i in enumerate(chosen)]
