"""Canonical contracts and fail-closed scientific data checks. No remote calls."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import jsonschema
import yaml

ROOT = Path(__file__).resolve().parent
LABELS = ['triage', 'preconsult', 'info', 'other']
ALIASES = {**dict(zip(['GUIDANCE','PRECONSULT','SERVICE','OTHER'],LABELS)), **{x:x for x in LABELS}}
TRAIN = {'train_gold','train_unlabeled','train_pseudo','train_synthetic'}
DEV = {'dev_tune','dev_calibration','dev_policy'}
SCHEMA = json.loads((ROOT/'config/annotation.schema.json').read_text(encoding='utf-8'))
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA, format_checker=jsonschema.FormatChecker())


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def write_json(path, value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def write_jsonl(path, values):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in values),encoding='utf-8')


def protocol():
    return yaml.safe_load((ROOT/'config/protocol.v1.yaml').read_text(encoding='utf-8'))


def hash_inputs(inputs):
    return hashlib.sha256(json.dumps(inputs,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def class_vector(values):
    if not isinstance(values,dict):raise ValueError('complete class probability map required')
    mapped={}
    for key,value in values.items():
        if key not in ALIASES:raise ValueError('unknown probability class')
        label=ALIASES[key]
        if label in mapped:raise ValueError('duplicate aliases cannot represent separate classes')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not 0<=value<=1:
            raise ValueError('invalid probability')
        mapped[label]=float(value)
    if set(mapped)!=set(LABELS):raise ValueError('missing class: never zero-fill top-k values')
    if abs(sum(mapped.values())-1)>1e-6:raise ValueError('probabilities do not sum to one')
    return [mapped[label] for label in LABELS]


def validate_record(row):
    VALIDATOR.validate(row)
    a=row['annotation'];inputs=row['inputs']
    if row['source']['kind'] in {'hospital_real','public_licensed'} and row['partition']!='unassigned' and row['source']['rights_status']!='approved':
        raise ValueError('rights approval must be recorded before real/licensed data enter a research partition')
    if '\ufffd' in inputs['user_input']:raise ValueError('replacement characters in input')
    if a['primary_intent'] is not None and a['primary_intent'] in a['secondary_intents']:
        raise ValueError('primary intent duplicated as secondary')
    if a['review_status'] in {'human_approved','audited_pseudo'}:
        if a['primary_intent'] is None and not (a['needs_clarification'] and a['policy_action']=='CLARIFY'):
            raise ValueError('unresolved primary must have a genuine clarification action')
        if a['review_status']=='human_approved' and len(a['annotators'])<2:
            raise ValueError('human gold needs two independent annotators/adjudication provenance')
    if a['risk_trigger'] is True:
        if a['policy_action']!='ESCALATE':raise ValueError('risk cannot be overridden by ordinary routing')
        if a['review_status']!='pending' and a['risk_review_status']!='clinician_approved':
            raise ValueError('approved safety annotation needs clinical review')
    if a['policy_action']=='ROUTE' and a['primary_intent'] is None:
        raise ValueError('cannot route an unresolved intent')
    if a['needs_clarification'] is True and a['policy_action'] not in {'CLARIFY','ESCALATE',None}:
        raise ValueError('clarification need conflicts with action')
    for evidence in a['evidence']:
        field=evidence['field']
        if field=='user_input':text=inputs['user_input']
        elif field.startswith('history.') and field.endswith('.content'):
            text=inputs['history'][int(field.split('.')[1])]['content']
        else:raise ValueError('unsupported evidence field')
        if evidence['start']>=evidence['end'] or text[evidence['start']:evidence['end']]!=evidence['text']:
            raise ValueError('evidence must match literal source with Unicode-codepoint offsets')
    for teacher in row['teacher_outputs']:
        if teacher['input_hash']!=hash_inputs(inputs):raise ValueError('teacher output belongs to different inputs')
        if teacher['probabilities'] is not None:class_vector(teacher['probabilities'])
        if teacher['eligible_for_soft_kd'] and not (teacher['probability_semantics']=='class_vector_structure_verified'
                and teacher['probabilities'] is not None
                and teacher['calibration_status']=='calibrated_on_training_gold'
                and teacher['teacher_policy_version']==row['policy_version']):
            raise ValueError('soft target is not calibrated or taxonomy-compatible')
    if row['partition'] in {'test_external','test_temporal'}:
        if row['source']['kind']!='hospital_real':raise ValueError('synthetic data cannot establish a real external/temporal test')
        if row['partition']=='test_external' and not row['grouping']['site_id']:raise ValueError('external test needs actual site id')
        if row['partition']=='test_temporal' and not row['grouping']['occurred_at']:raise ValueError('temporal test needs observed time')
    return row


def dataset_errors(rows):
    errors=[];seen=set()
    for row in rows:
        ident=row.get('record_id')
        if ident in seen:errors.append({'record_id':ident,'error':'duplicate_id'})
        seen.add(ident)
        try:validate_record(row)
        except (ValueError,KeyError,IndexError,jsonschema.ValidationError) as error:
            errors.append({'record_id':ident,'error':str(error).split('\n')[0]})
    return errors


def assert_teacher_access(row, purpose='label_training'):
    part=row['partition']
    if purpose in {'label_training','synthesize','active_select','fit_teacher_calibration'} and (part in DEV or part.startswith('test') or part=='unassigned'):
        raise ValueError('heldout/unassigned content cannot enter training teacher requests')
    if row['source']['kind']=='hospital_real' and not row['source']['external_transfer_authorized']:
        raise ValueError('external transfer authorization not recorded')
    if purpose=='fit_teacher_calibration' and (part!='train_gold' or row['annotation']['review_status']!='human_approved'):
        raise ValueError('teacher calibration requires training-side human gold')
    return True


def leakage_audit(rows, settings=None):
    settings=settings or protocol()['split_policy']
    owners={};findings=[]
    for row in rows:
        part=row['partition']
        if part=='unassigned':continue
        fold='train' if part in TRAIN else part
        grouping=row['grouping']
        keys=[('exact_input',hash_inputs(row['inputs']))]+[(key,grouping.get(key)) for key in settings['isolate_groups'] if grouping.get(key)]
        for key in keys:
            old=owners.setdefault(key,(fold,row['record_id']))
            if old[0]!=fold:findings.append({'type':key[0],'left':old,'right':[fold,row['record_id']]})
        if grouping.get('site_id') in settings.get('external_site_ids',[]) and part!='test_external':
            findings.append({'type':'external_site_contamination','record_id':row['record_id']})
        cutoff=settings.get('temporal_cutoff_utc')
        if cutoff and grouping.get('occurred_at'):
            stamp=datetime.fromisoformat(grouping['occurred_at'].replace('Z','+00:00'))
            boundary=datetime.fromisoformat(cutoff.replace('Z','+00:00'))
            if stamp>=boundary and part in TRAIN|DEV:
                findings.append({'type':'future_data_in_training','record_id':row['record_id']})
            if stamp<boundary and part=='test_temporal':
                findings.append({'type':'temporal_test_before_cutoff','record_id':row['record_id']})
    return findings


def migrate_handoff(path):
    rows=[]
    for old in read_jsonl(path):
        source_kind='assistant_synthetic' if old['source']=='assistant_authored' else 'llm_synthetic'
        inputs={k:old[k] for k in ['user_input','history','context']}
        teachers=[]
        for name,answer in old.get('teacher_annotations',{}).items():
            probabilities=answer.get('answer',{}).get('probabilities')
            semantics='hard_label_only'
            if probabilities is not None:
                try:class_vector(probabilities);semantics='class_vector_structure_verified'
                except ValueError:probabilities=None;semantics='invalid_not_for_training'
            teachers.append({'teacher_id':name,'model_version':answer.get('teacher_model') or answer.get('response_model'),
                'teacher_policy_version':answer['teacher_policy_version'],'prediction':answer.get('prediction'),
                'probabilities':probabilities,'probability_semantics':semantics,'calibration_status':'not_calibrated',
                'eligible_for_soft_kd':False,'input_hash':hash_inputs(inputs),'reference_response':answer})
        rows.append({'record_id':'care-'+hashlib.sha256(old['id'].encode()).hexdigest()[:16],
            'policy_version':'care-prep-v1','partition':'unassigned',
            'grouping':{'family_id':old['family_id'],'session_group_id':None,'patient_group_id':None,'site_id':None,
                        'occurred_at':None,'generator_seed_group_id':old.get('scenario_group'),'semantic_cluster_id':None},
            'source':{'kind':source_kind,'rights_status':'synthetic_no_patient_source','external_transfer_authorized':True},
            'inputs':inputs,'annotation':{'primary_intent':None,'secondary_intents':[],'acceptable_intents':[],
                'needs_clarification':None,'policy_action':None,'risk_trigger':None,'risk_review_status':'not_assessed',
                'review_status':'pending','annotators':[],'evidence':[]},'teacher_outputs':teachers,
            'provenance':{'source_record_id':old['id'],'source_policy_version':old['policy_version'],
                'proposed_primary_intent_not_gold':old.get('proposed_label',old.get('label')),
                'source_tags':old.get('tags',[]),'requires_new_policy_review':True}})
    return rows
