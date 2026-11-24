"""Offline experiment preparation CLI. Never downloads weights, trains or calls teachers."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import random
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from care_contract import ROOT, LABELS, TRAIN, protocol, read_jsonl, write_json, write_jsonl, migrate_handoff, dataset_errors, leakage_audit, hash_inputs
from care_metrics import evaluate, fit_temperature, fit_policy, apply_policy
from care_statistics import paired_comparison, holm, safety_sample_size_report
from care_acquisition import select


def load_predictions(path, rows):
    values=read_jsonl(path);by_id={r['record_id']:r for r in values}
    if len(by_id)!=len(values):raise ValueError('duplicate prediction ids')
    expected={r['record_id']:r for r in rows}
    if set(by_id)-set(expected):raise ValueError('predictions contain records outside frozen input set')
    for key,pred in by_id.items():
        if pred.get('input_hash')!=hash_inputs(expected[key]['inputs']):raise ValueError('stale input fingerprint')
    return by_id


def near_duplicates(rows, threshold=.92):
    if len(rows)<2:return []
    texts=[json.dumps(r['inputs'],ensure_ascii=False,sort_keys=True) for r in rows]
    x=TfidfVectorizer(analyzer='char',ngram_range=(2,4),max_features=40000).fit_transform(texts)
    pairs=[]
    # Chunk the multiplication; never materialize an entire 18k by 18k dense matrix.
    for start in range(0,len(rows),128):
        similarities=(x[start:start+128]@x.T).tocoo()
        for i,j,value in zip(similarities.row,similarities.col,similarities.data):
            i+=start
            if i<j and value>=threshold:
                pairs.append({'left':rows[i]['record_id'],'right':rows[j]['record_id'],'lexical_context_similarity':float(value),
                              'same_family':rows[i]['grouping']['family_id']==rows[j]['grouping']['family_id'],
                              'decision':'human_review_not_automatic_delete'})
    return pairs


def make_matrix(settings):
    rows=[]
    for variant in settings['experiment_scope']['primary_variants']:
        for seed in settings['experiment_scope']['train_seeds']:
            teacher_count=0 if variant=='gold_only' else 2 if variant=='heterogeneous_audited' else 1
            budget=0 if not teacher_count else 2000
            rows.append({'run_id':f'qwen3-8b_{variant}_seed{seed}','student':'Qwen3-8B-text',
                'variant':variant,'seed':seed,'human_gold_train':2000,'teacher_judgment_cap':budget,
                'max_candidates_before_qa':budget//teacher_count if teacher_count else 0,
                'additional_human_review_budget':0,'disagreements_without_budget':'quarantine_not_force_label',
                'same_training_updates_as_matched_control':True,'status':'planned_blocked_until_gold_and_model_available',
                'teacher_votes_are_not_free':True})
    return rows


def build(args):
    input_sha=hashlib.sha256(Path(args.handoff).read_bytes()).hexdigest()
    existing_readiness=ROOT/'reports/readiness.json'
    if existing_readiness.exists() and json.loads(existing_readiness.read_text(encoding='utf-8')).get('source_handoff_sha256')!=input_sha:
        raise ValueError('source handoff changed; create a new preparation version instead of overwriting annotation inputs')
    rows=migrate_handoff(args.handoff)
    errors=dataset_errors(rows)
    if errors:raise ValueError(str(errors[:3]))
    write_jsonl(ROOT/'data/candidates.canonical.jsonl',rows)
    pairs=near_duplicates(rows)
    write_json(ROOT/'reports/near_duplicate_candidates.json',{'threshold':.92,'pairs':pairs,
        'method':'context-aware lexical character TF-IDF; not an embedding semantic audit'})
    settings=protocol();counts=Counter(r['partition'] for r in rows)
    matrix=make_matrix(settings)
    write_json(ROOT/'plans/run_matrix.json',matrix)
    write_json(ROOT/'plans/results_registry.template.json',[{**job,'actual_train_n':None,'model_run_id':None,'metrics':None,'predictions_sha256':None,'actual_cost':None,'failure_reason':None} for job in matrix])
    write_json(ROOT/'reports/safety_sample_size.json',safety_sample_size_report())
    constraints={
        'candidate_schema_passed':not errors,
        'human_gold_available':any(r['annotation']['review_status']=='human_approved' for r in rows),
        'gold_target_6000_reached':sum(r['annotation']['review_status']=='human_approved' for r in rows)>=6000,
        'actual_external_site_selected':bool(settings['split_policy']['external_site_ids']),
        'temporal_cutoff_selected':bool(settings['split_policy']['temporal_cutoff_utc']),
        'clinical_safety_policy_approved':settings['required_human_decisions']['clinical_safety_policy']=='approved',
        'training_model_preparation_owned_by_user':True}
    readiness={'status':'OFFLINE_PREPARATION_READY_RESEARCH_NOT_READY_TO_RUN',
        'candidates':len(rows),'human_approved':0,'partition_counts':dict(counts),'checks':constraints,
        'blocking_research_prerequisites':[k for k,v in constraints.items() if not v],
        'teacher_class_vectors_structure_verified':sum(t['probabilities'] is not None for r in rows for t in r['teacher_outputs']),
        'eligible_calibrated_soft_targets':sum(t['eligible_for_soft_kd'] for r in rows for t in r['teacher_outputs']),
        'live_api_calls_this_build':0,'training_runs_started':0,
        'source_handoff_sha256':hashlib.sha256(Path(args.handoff).read_bytes()).hexdigest()}
    for owner,seed in [('you',101),('nayang',202)]:
        packet_path=ROOT/f'data/annotation_{owner}.v1.jsonl'
        if not packet_path.exists():
            packet=[{'record_id':r['record_id'],'policy_version':r['policy_version'],
                     'family_id':'family-'+hashlib.sha256(r['grouping']['family_id'].encode()).hexdigest()[:12],
                     'inputs':copy.deepcopy(r['inputs']),'input_hash':hash_inputs(r['inputs']),
                     'annotation':copy.deepcopy(r['annotation']),'comment':''} for r in rows]
            random.Random(seed).shuffle(packet);write_jsonl(packet_path,packet)
    write_json(ROOT/'reports/readiness.json',readiness)
    versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn','jsonschema','PyYAML','requests','beautifulsoup4']}
    write_json(ROOT/'reports/environment.json',{'python_packages':versions,'gpu_or_training_environment_checked':False})
    source_hashes={str(p.relative_to(ROOT/'sources')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'sources').rglob('*') if p.is_file() and p!=ROOT/'sources/manifest.json'}
    write_json(ROOT/'sources/manifest.json',{'files':source_hashes,'shared_chat_limit':'visible transcript captured; first and last assistant bodies unavailable'})
    print(json.dumps(readiness,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('build');q.add_argument('--handoff',default=str(ROOT.parent/'data/handoff_v0.2/candidates_with_teachers.jsonl'))
    q=sub.add_parser('validate');q.add_argument('--input',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('evaluate');q.add_argument('--gold',required=True);q.add_argument('--predictions',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('calibrate');q.add_argument('--gold',required=True);q.add_argument('--predictions',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('fit-policy');q.add_argument('--gold',required=True);q.add_argument('--predictions',required=True);q.add_argument('--calibration',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('apply-policy');q.add_argument('--input',required=True);q.add_argument('--predictions',required=True);q.add_argument('--calibration',required=True);q.add_argument('--policy',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('select');q.add_argument('--pool',required=True);q.add_argument('--predictions');q.add_argument('--method',choices=['random','margin','hybrid'],default='hybrid');q.add_argument('--budget',type=int,required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('compare');q.add_argument('--gold',required=True);q.add_argument('--left',nargs='+',required=True);q.add_argument('--right',nargs='+',required=True);q.add_argument('--seeds',type=int,nargs='+',required=True);q.add_argument('--replicates',type=int,default=10000);q.add_argument('--output',required=True)
    q=sub.add_parser('holm');q.add_argument('--pvalues',required=True);q.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='build':return build(a)
    if a.command=='validate':
        rows=read_jsonl(a.input);errors=dataset_errors(rows);leaks=leakage_audit(rows)
        write_json(a.output,{'schema_errors':errors,'cross_split_leaks':leaks,'valid':not errors and not leaks})
        if errors or leaks:raise SystemExit(2)
        return
    if a.command in {'evaluate','calibrate','fit-policy','compare'}:
        rows=read_jsonl(a.gold);errors=dataset_errors(rows)
        if errors:raise ValueError(str(errors[:2]))
    if a.command=='evaluate':result=evaluate(rows,load_predictions(a.predictions,rows))
    elif a.command=='calibrate':result=fit_temperature(rows,load_predictions(a.predictions,rows))
    elif a.command=='fit-policy':
        calibration=json.loads(Path(a.calibration).read_text(encoding='utf-8'))
        predictions=load_predictions(a.predictions,rows)
        if {p.get('model_run_id') for p in predictions.values()}!={calibration.get('model_run_id')}:raise ValueError('policy fold checkpoint differs from calibration checkpoint')
        result=fit_policy(rows,predictions,calibration['temperature'],protocol()['policy_threshold_fitting'])
        result['model_run_id']=calibration['model_run_id']
    elif a.command=='apply-policy':
        rows=read_jsonl(a.input)
        result=apply_policy(rows,load_predictions(a.predictions,rows),json.loads(Path(a.calibration).read_text(encoding='utf-8')),json.loads(Path(a.policy).read_text(encoding='utf-8')))
        write_jsonl(a.output,result);return
    elif a.command=='select':
        rows=read_jsonl(a.pool)
        errors=dataset_errors(rows)
        if errors:raise ValueError(str(errors[:2]))
        predictions=load_predictions(a.predictions,rows) if a.predictions else None
        result=select(rows,a.budget,a.method,predictions);write_jsonl(a.output,result);return
    elif a.command=='compare':
        result=paired_comparison(rows,[load_predictions(f,rows) for f in a.left],
            [load_predictions(f,rows) for f in a.right],a.seeds,a.replicates)
    elif a.command=='holm':result=holm(json.loads(Path(a.pvalues).read_text(encoding='utf-8')))
    write_json(a.output,result)


if __name__=='__main__':main()
