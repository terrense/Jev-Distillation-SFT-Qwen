"""Join independently completed research annotations; never infer medical approval."""
import argparse
import copy
from collections import Counter
from care_contract import read_jsonl,write_jsonl,write_json,hash_inputs,validate_record


def merge(rows,left,right,left_name='you',right_name='nayang'):
    if left_name==right_name:raise ValueError('independent annotators required')
    ids={r['record_id'] for r in rows}
    def index(values):
        result={r['record_id']:r for r in values}
        if len(result)!=len(values) or set(result)!=ids:raise ValueError('annotation packet ids mismatch')
        return result
    a,b=index(left),index(right);accepted=[];unresolved=[];votes=[]
    def decision_key(annotation):
        return (annotation['primary_intent'],tuple(sorted(annotation['secondary_intents'])),
                tuple(sorted(annotation['acceptable_intents'])),annotation['needs_clarification'],
                annotation['policy_action'],annotation['risk_trigger'])
    for row in rows:
        x,y=a[row['record_id']],b[row['record_id']]
        for packet in [x,y]:
            if packet['policy_version']!=row['policy_version'] or packet['inputs']!=row['inputs'] or packet['input_hash']!=hash_inputs(row['inputs']):
                raise ValueError('input or policy changed; re-label after versioning')
        aa,bb=x['annotation'],y['annotation'];item=copy.deepcopy(row)
        item['provenance']['review_votes']={left_name:aa,right_name:bb}
        item['provenance']['review_comments']=[x.get('comment',''),y.get('comment','')]
        if aa['primary_intent'] is not None and bb['primary_intent'] is not None:votes.append((aa['primary_intent'],bb['primary_intent']))
        completed=(aa['primary_intent'] is not None or aa['needs_clarification'] is True) and aa['policy_action'] is not None
        if not completed or decision_key(aa)!=decision_key(bb):
            item['provenance']['adjudication_reason']='incomplete_or_disagreement';unresolved.append(item);continue
        if aa['risk_trigger'] is True or aa['policy_action']=='ESCALATE':
            item['provenance']['adjudication_reason']='independent_clinical_or_escalation_review_required';unresolved.append(item);continue
        annotation=copy.deepcopy(aa);annotation.update(review_status='human_approved',annotators=[left_name,right_name],risk_review_status='ops_screened' if aa['risk_trigger'] is False else 'not_assessed')
        evidence={tuple(sorted(e.items())):e for e in aa['evidence']+bb['evidence']}
        annotation['evidence']=list(evidence.values());item['annotation']=annotation
        validate_record(item);accepted.append(item)
    kappa=None;agreement=None
    if votes:
        n=len(votes);agreement=sum(x==y for x,y in votes)/n
        ca,cb=Counter(x for x,_ in votes),Counter(y for _,y in votes)
        expected=sum(ca[k]*cb[k] for k in ca)/n**2
        kappa=(agreement-expected)/(1-expected) if expected<1 else None
    return accepted,unresolved,{'records':len(rows),'accepted':len(accepted),'unresolved':len(unresolved),
        'jointly_class_labeled':len(votes),'class_agreement':agreement,'class_cohen_kappa':kappa,
        'agreement_is_not_a_medical_safety_validation':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',default='data/candidates.canonical.jsonl');p.add_argument('--left',required=True);p.add_argument('--right',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();accepted,unresolved,stats=merge(read_jsonl(a.candidates),read_jsonl(a.left),read_jsonl(a.right))
    from pathlib import Path
    dest=Path(a.output);write_jsonl(dest/'human_approved.jsonl',accepted);write_jsonl(dest/'unresolved.jsonl',unresolved);write_json(dest/'agreement.json',stats)
    print(stats)


if __name__=='__main__':main()
