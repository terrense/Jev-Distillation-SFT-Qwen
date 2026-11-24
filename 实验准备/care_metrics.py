"""Explicit four-class, action and selective-prediction metrics. No model/API calls."""
import math
from collections import Counter
import numpy as np
from scipy.optimize import minimize_scalar
from care_contract import LABELS, class_vector, hash_inputs


def confusion_and_f1(truth, predictions):
    matrix=np.zeros((4,5),dtype=int)
    for y,p in zip(truth,predictions):matrix[LABELS.index(y),LABELS.index(p) if p in LABELS else 4]+=1
    return matrix, f1_from_confusion(matrix)


def f1_from_confusion(matrix):
    matrix=np.asarray(matrix,dtype=float)
    tp=np.diag(matrix[:,:4]);support=matrix.sum(axis=1);pred=matrix[:,:4].sum(axis=0)
    precision=np.divide(tp,pred,out=np.zeros(4),where=pred>0)
    recall=np.divide(tp,support,out=np.zeros(4),where=support>0)
    f1=np.divide(2*tp,pred+support,out=np.zeros(4),where=(pred+support)>0)
    return {'macro_f1':float(np.mean(f1)) if support.sum() else None,'accuracy':float(tp.sum()/support.sum()) if support.sum() else None,
            'supported_macro_f1':float(np.mean(f1[support>0])) if any(support>0) else None,
            'per_class':{label:{'support':int(support[i]),'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i])} for i,label in enumerate(LABELS)}}


def probability_metrics(truth, probabilities, bins=10):
    if not truth:return {'n':0,'brier':None,'ece':None,'nll':None,'aurc':None,'risk_at_coverage':[]}
    p=np.asarray(probabilities,dtype=float); y=np.array([LABELS.index(x) for x in truth]);n=len(y)
    one_hot=np.eye(4)[y]
    confidence=p.max(axis=1); correct=p.argmax(axis=1)==y
    ece=0.;reliability=[]
    indices=np.minimum((confidence*bins).astype(int),bins-1)
    for b in range(bins):
        mask=indices==b
        if mask.any():
            acc=float(correct[mask].mean());conf=float(confidence[mask].mean())
            ece+=mask.mean()*abs(acc-conf)
            reliability.append({'bin':b,'n':int(mask.sum()),'confidence':conf,'accuracy':acc})
    # A threshold accepts complete confidence ties. Never use correctness to break ties.
    cumulative=0;errors=0;curve=[];expected_risks=[]
    for conf in sorted(set(confidence),reverse=True):
        mask=confidence==conf;m=int(mask.sum());bad=int((~correct[mask]).sum())
        for j in range(1,m+1):expected_risks.append((errors+j*bad/m)/(cumulative+j))
        cumulative+=m;errors+=bad
        curve.append({'threshold':float(conf),'coverage':cumulative/n,'risk':errors/cumulative})
    targets=[]
    for target in [.8,.9,.95]:
        allowed=[x for x in curve if x['coverage']<=target+1e-12]
        point=allowed[-1] if allowed else {'coverage':0.,'risk':None,'threshold':None}
        targets.append({'target_coverage':target,'achieved_coverage':point['coverage'],'risk':point['risk'],'threshold':point['threshold']})
    return {'n':n,'brier':float(((p-one_hot)**2).sum(axis=1).mean()),
            'nll':float(-np.log(np.clip(p[np.arange(n),y],1e-12,1)).mean()),'ece':float(ece),
            'ece_bins':bins,'reliability':reliability,'aurc':float(np.mean(expected_risks)),
            'aurc_tie_rule':'expected_within_tie_order_not_label_sorted','risk_at_coverage':targets,'risk_coverage_curve':curve}


def fit_temperature(rows, predictions):
    truth=[];vectors=[]
    run_ids={p.get('model_run_id') for p in predictions.values()}
    if len(run_ids)!=1 or None in run_ids:raise ValueError('one explicit checkpoint/model_run_id required for calibration')
    for row in rows:
        a=row['annotation']
        if row['partition']!='dev_calibration' or a['review_status']!='human_approved':
            raise ValueError('temperature fitting requires approved dev_calibration records only')
        if a['primary_intent'] is None:continue
        p=predictions.get(row['record_id'],{})
        if p.get('error') or not p.get('format_valid',True):raise ValueError('invalid calibration prediction')
        vectors.append(class_vector(p.get('probabilities')));truth.append(LABELS.index(a['primary_intent']))
    if not truth or len(set(truth))!=4:raise ValueError('calibration fold must include all four classes')
    logp=np.log(np.clip(np.asarray(vectors),1e-12,1));y=np.array(truth)
    def loss(log_t):
        z=logp/math.exp(log_t);z-=z.max(axis=1,keepdims=True)
        q=np.exp(z);q/=q.sum(axis=1,keepdims=True)
        return float(-np.log(np.clip(q[np.arange(len(y)),y],1e-12,1)).mean())
    result=minimize_scalar(loss,bounds=(math.log(.05),math.log(20)),method='bounded')
    return {'temperature':math.exp(float(result.x)),'calibration_n':len(truth),'nll_before':loss(0),'nll_after':float(result.fun),
            'scope':'class_probabilities_only_not_token_logits','label_order':LABELS,'model_run_id':next(iter(run_ids)),'log_probability_clip':1e-12,
            'note':'Clipped probability temperature is a declared approximation; it cannot recover hidden unrounded logits.'}


def apply_temperature(probabilities, temperature):
    if temperature<=0:raise ValueError('temperature must be positive')
    values=np.log(np.clip(np.asarray(class_vector(probabilities)),1e-12,1))/temperature
    values-=values.max();p=np.exp(values);p/=p.sum()
    return dict(zip(LABELS,map(float,p)))


def evaluate(rows, predictions):
    gold=[r for r in rows if r['annotation']['review_status']=='human_approved']
    if len(gold)!=len(rows):raise ValueError('formal evaluation requires human-approved gold; do not auto-promote candidates')
    known=[r for r in rows if r['annotation']['primary_intent'] is not None]
    truth=[];guesses=[];prob_truth=[];probabilities=[];errors=Counter();latencies=[]
    action_correct=0;action_n=0;workflow_correct=0;workflow_n=0
    clarify_tp=clarify_fp=clarify_fn=0;safety_tp=safety_fn=safety_fp=safety_neg=0
    valid_routes=correct_routes=0;secondary_tp=secondary_fp=secondary_fn=secondary_exact=secondary_n=0
    for row in rows:
        a=row['annotation']; pred=predictions.get(row['record_id'],{})
        valid=bool(pred) and not pred.get('error') and pred.get('format_valid',True) and pred.get('prediction') in LABELS
        legal_abstention=bool(pred) and not pred.get('error') and pred.get('format_valid',True) and pred.get('prediction') is None and pred.get('action') in {'CLARIFY','ESCALATE'}
        if not pred:errors['missing_prediction']+=1
        elif pred.get('error'):errors['dependency_error']+=1
        elif not valid and not legal_abstention:errors['format_or_label_error']+=1
        p=pred.get('prediction') if valid else None
        if a['primary_intent'] is not None:
            truth.append(a['primary_intent']);guesses.append(p)
            if valid and pred.get('probabilities') is not None:
                try:
                    vector=class_vector(pred['probabilities']);prob_truth.append(a['primary_intent']);probabilities.append(vector)
                except ValueError:errors['invalid_probability_vector']+=1
        action=pred.get('action') if pred and not pred.get('error') else None
        if a['policy_action'] is not None:
            action_n+=1;action_correct+=action==a['policy_action']
            expected=a['policy_action']=='CLARIFY';actual=action=='CLARIFY'
            clarify_tp+=expected and actual;clarify_fp+=not expected and actual;clarify_fn+=expected and not actual
            workflow_n+=1
            workflow_correct+=(action==a['policy_action'] and (action!='ROUTE' or p==a['primary_intent']))
        if action=='ROUTE':
            valid_routes+=1;correct_routes+=a['policy_action']=='ROUTE' and p==a['primary_intent']
        if a['risk_review_status']=='clinician_approved' and a['risk_trigger'] is not None:
            if a['risk_trigger']:
                safety_tp+=action=='ESCALATE';safety_fn+=action!='ESCALATE'
            else:
                safety_neg+=1;safety_fp+=action=='ESCALATE'
        if pred.get('secondary_intents') is not None:
            expected=set(a['secondary_intents']);actual=set(pred['secondary_intents'])
            secondary_n+=1;secondary_exact+=expected==actual
            secondary_tp+=len(expected&actual);secondary_fp+=len(actual-expected);secondary_fn+=len(expected-actual)
        if isinstance(pred.get('latency_ms'),(int,float)):latencies.append(pred['latency_ms'])
    confusion,classification=confusion_and_f1(truth,guesses)
    pm=probability_metrics(prob_truth,probabilities)
    def ratio(a,b):return a/b if b else None
    in_scope=sum(1 for y in truth if y!='other');out_scope=sum(1 for y in truth if y=='other')
    return {'n_records':len(rows),'n_class_resolvable':len(known),'n_unresolved_primary':len(rows)-len(known),
        **classification,'confusion_columns':LABELS+['ERROR'],'confusion':confusion.tolist(),
        'false_rejection_rate':ratio(sum(y!='other' and p=='other' for y,p in zip(truth,guesses)),in_scope),
        'out_of_scope_acceptance_rate':ratio(sum(y=='other' and p in LABELS and p!='other' for y,p in zip(truth,guesses)),out_scope),
        'error_counts':dict(errors),'action_accuracy':ratio(action_correct,action_n),
        'workflow_activation_accuracy':ratio(workflow_correct,workflow_n),
        'clarify_precision':ratio(clarify_tp,clarify_tp+clarify_fp),'clarify_recall':ratio(clarify_tp,clarify_tp+clarify_fn),
        'route_accuracy_conditional_on_route':ratio(correct_routes,valid_routes),
        'safety_sensitivity':ratio(safety_tp,safety_tp+safety_fn),'clinician_reviewed_risk_positive_n':safety_tp+safety_fn,
        'over_escalation_rate':ratio(safety_fp,safety_neg),'clinician_reviewed_risk_negative_n':safety_neg,
        'secondary_exact_match':ratio(secondary_exact,secondary_n),
        'secondary_micro_f1':ratio(2*secondary_tp,2*secondary_tp+secondary_fp+secondary_fn),
        'probability_coverage':ratio(len(probabilities),len(known)),'probability_metrics':pm,
        'latency_p50_ms':float(np.quantile(latencies,.5)) if latencies else None,
        'latency_p95_ms':float(np.quantile(latencies,.95)) if latencies else None}


def decision(prediction, probability_map, safety_flag, threshold, margin):
    if safety_flag:return {'action':'ESCALATE','prediction':prediction,'reason_code':'independent_safety_policy'}
    values=sorted(class_vector(probability_map),reverse=True)
    if values[0]<threshold or values[0]-values[1]<margin:
        return {'action':'CLARIFY','prediction':prediction,'reason_code':'calibrated_uncertainty'}
    if prediction not in LABELS:raise ValueError('cannot route malformed class')
    return {'action':'ROUTE','prediction':prediction,'handler':'scope_refusal' if prediction=='other' else prediction,'reason_code':'within_locked_thresholds'}


def fit_policy(rows, predictions, temperature, settings):
    if not rows:raise ValueError('empty policy fold')
    for row in rows:
        if row['partition']!='dev_policy' or row['annotation']['review_status']!='human_approved':
            raise ValueError('policy tuning requires approved dev_policy only')
        if row['annotation']['policy_action'] is None:raise ValueError('gold action labels required')
        pred=predictions.get(row['record_id'],{})
        if pred.get('error') or not pred.get('format_valid',True):raise ValueError('valid predictions required')
        class_vector(pred.get('probabilities'))
    costs=settings['candidate_research_costs'];candidates=[]
    for threshold in settings['thresholds']:
        for margin in settings['margins']:
            losses=[];routes=0
            for row in rows:
                gold=row['annotation'];pred=predictions[row['record_id']]
                calibrated=apply_temperature(pred['probabilities'],temperature)
                label=max(calibrated,key=calibrated.get)
                # Gold risk is used for scoring only; never supplied as an inference feature.
                output=decision(label,calibrated,pred.get('independent_safety_flag') is True,threshold,margin)
                action=output['action'];routes+=action=='ROUTE'
                if gold['policy_action']=='ESCALATE' and action!='ESCALATE':loss=costs['missed_escalation']
                elif action=='ESCALATE' and gold['policy_action']!='ESCALATE':loss=costs['unnecessary_escalation']
                elif action=='ROUTE' and (gold['policy_action']!='ROUTE' or label!=gold['primary_intent']):loss=costs['wrong_route']
                elif action=='CLARIFY' and gold['policy_action']=='ROUTE':loss=costs['unnecessary_clarify']
                else:loss=0
                losses.append(loss)
            candidates.append({'threshold':threshold,'margin':margin,'mean_research_cost':float(np.mean(losses)),'route_fraction':routes/len(rows)})
    best=min(candidates,key=lambda x:(x['mean_research_cost'],-x['route_fraction'],x['threshold'],x['margin']))
    return {'selected':best,'grid':candidates,'n':len(rows),'temperature':temperature,
            'costs':costs,'clinical_utility_approved':False,'status':'research_candidate_not_deployment_approval',
            'safety_gate_input':'independent_safety_flag_from_prediction_not_gold_risk'}


def apply_policy(rows, predictions, calibration, policy):
    if calibration.get('label_order')!=LABELS:raise ValueError('calibration label order mismatch')
    if policy.get('model_run_id')!=calibration.get('model_run_id'):raise ValueError('policy/calibration model binding mismatch')
    result=[]
    for row in rows:
        pred=predictions.get(row['record_id'],{})
        if pred.get('error') or not pred or not pred.get('format_valid',True):
            result.append({'record_id':row['record_id'],'input_hash':hash_inputs(row['inputs']),'error':pred.get('error','MISSING_OR_INVALID_PREDICTION')});continue
        if pred.get('model_run_id')!=calibration['model_run_id']:raise ValueError('cannot apply one checkpoint temperature to another')
        calibrated=apply_temperature(pred['probabilities'],calibration['temperature'])
        output=decision(max(calibrated,key=calibrated.get),calibrated,pred.get('independent_safety_flag') is True,
                        policy['selected']['threshold'],policy['selected']['margin'])
        result.append({**pred,**output,'probabilities':calibrated,'raw_prediction':pred.get('prediction'),
                       'calibration_temperature':calibration['temperature'],
                       'independent_safety_policy_available':bool(pred.get('safety_policy_version')),
                       'research_policy_not_clinical_deployment_approval':True})
    return result
