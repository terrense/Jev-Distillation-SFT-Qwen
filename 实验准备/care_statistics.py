"""Paired seed/group uncertainty and explicit multiplicity correction."""
import math
import numpy as np
from care_contract import LABELS
from care_metrics import f1_from_confusion


def holm(pvalues):
    ordered=sorted(pvalues,key=pvalues.get);m=len(ordered);adjusted={};last=0.
    for rank,key in enumerate(ordered):
        value=pvalues[key]
        if not 0<=value<=1:raise ValueError('p-value outside [0,1]')
        last=max(last,min(1.,(m-rank)*value));adjusted[key]=last
    return adjusted


def cluster_ids(rows):
    parent=list(range(len(rows)));owners={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(rows):
        for key in ['family_id','patient_group_id','session_group_id','generator_seed_group_id','semantic_cluster_id']:
            value=row['grouping'].get(key)
            if value is not None:
                token=(key,value)
                if token in owners:parent[find(i)]=find(owners[token])
                else:owners[token]=i
    roots={root:j for j,root in enumerate(sorted({find(i) for i in range(len(rows))}))}
    return [roots[find(i)] for i in range(len(rows))]


def paired_comparison(rows, left_by_seed, right_by_seed, seeds, replicates=10000, rng_seed=20261118):
    if len(seeds)!=len(left_by_seed) or len(seeds)!=len(right_by_seed) or len(set(seeds))!=len(seeds):
        raise ValueError('both methods require the same distinct paired seeds')
    if any(r['annotation']['review_status']!='human_approved' for r in rows):raise ValueError('human gold required for every evaluation record')
    all_ids={r['record_id'] for r in rows}
    rows=[r for r in rows if r['annotation']['primary_intent'] is not None]
    if not rows:raise ValueError('no resolvable class labels')
    if any(r['annotation']['review_status']!='human_approved' for r in rows):raise ValueError('human gold required')
    groups=cluster_ids(rows);g=max(groups)+1;s=len(seeds)
    left=np.zeros((s,g,4,5));right=np.zeros_like(left)
    for k,(a,b) in enumerate(zip(left_by_seed,right_by_seed)):
        if (set(a)|set(b))-all_ids:
            raise ValueError('methods include outputs outside the same frozen records')
        for i,row in enumerate(rows):
            truth=LABELS.index(row['annotation']['primary_intent'])
            for values,dest in [(a,left),(b,right)]:
                pred=values.get(row['record_id'],{'error':'MISSING_PREDICTION'})
                value=pred.get('prediction') if not pred.get('error') and pred.get('format_valid',True) else None
                dest[k,groups[i],truth,LABELS.index(value) if value in LABELS else 4]+=1
    scores=lambda matrix:np.array([f1_from_confusion(m)['macro_f1'] for m in matrix])
    a_scores=scores(left.sum(axis=1));b_scores=scores(right.sum(axis=1));observed=float((a_scores-b_scores).mean())
    rng=np.random.default_rng(rng_seed);deltas=[];null=[]
    for _ in range(replicates):
        sampled_seeds=rng.integers(0,s,s);sampled_groups=rng.integers(0,g,g)
        a=left[sampled_seeds][:,sampled_groups].sum(axis=1)
        b=right[sampled_seeds][:,sampled_groups].sum(axis=1)
        deltas.append(float((scores(a)-scores(b)).mean()))
        # Swap each whole group across every seed, preserving within-group/seed dependence.
        swap=rng.integers(0,2,g).astype(bool)[None,:,None,None]
        pa=np.where(swap,right,left).sum(axis=1);pb=np.where(swap,left,right).sum(axis=1)
        null.append(float((scores(pa)-scores(pb)).mean()))
    return {'metric':'macro_f1','effect_direction':'left_minus_right','n_records':len(rows),'n_groups':g,
            'seeds':seeds,'left_per_seed':a_scores.tolist(),'right_per_seed':b_scores.tolist(),
            'left_missing_outputs':[len(all_ids-set(p)) for p in left_by_seed],
            'right_missing_outputs':[len(all_ids-set(p)) for p in right_by_seed],
            'left_mean':float(a_scores.mean()),'right_mean':float(b_scores.mean()),
            'left_seed_sd':float(a_scores.std(ddof=1)) if s>1 else None,
            'right_seed_sd':float(b_scores.std(ddof=1)) if s>1 else None,
            'delta':observed,'delta_ci95':np.quantile(deltas,[.025,.975]).tolist(),
            'ci_method':'paired_two_axis_seed_and_group_bootstrap',
            'cluster_permutation_p':float((1+sum(abs(x)>=abs(observed)-1e-12 for x in null))/(replicates+1)),
            'multiplicity_adjustment':'apply Holm across preregistered contrast family, not inside this single call',
            'replicates':replicates,'rng_seed':rng_seed,
            'warning':'single site does not identify between-site variance; five seeds remain a small training-variance sample'}


def safety_sample_size_report():
    return {'method':'one_sided_exact_95_percent_bound_when_zero_misses',
            'zero_miss_upper_rates':{str(n):1-.05**(1/n) for n in [100,200,299,600]},
            'positive_cases_required_with_zero_misses':{str(target):math.ceil(math.log(.05)/math.log(1-target)) for target in [.01,.005]},
            'planned_safety_slice':{'positive':100,'negative':100},
            'implication':'200 total with 100 positives cannot establish a miss rate below 1%, even with zero observed misses',
            'not_a_clinical_safety_guarantee':True}
