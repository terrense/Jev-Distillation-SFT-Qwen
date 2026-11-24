"""Software fixtures only. No real gold labels, clinical claims, model fitting or API calls."""
import copy
import unittest
from care_contract import ROOT,LABELS,class_vector,validate_record,hash_inputs,read_jsonl,assert_teacher_access,leakage_audit
from care_metrics import probability_metrics,evaluate,fit_temperature,fit_policy,apply_policy,decision
from care_statistics import holm,paired_comparison,safety_sample_size_report
from care_acquisition import select


def fixture(label='triage', ident='fixture-0', partition='train_gold'):
    row=copy.deepcopy(read_jsonl(ROOT/'data/candidates.canonical.jsonl')[0])
    row['record_id']=ident;row['partition']=partition;row['teacher_outputs']=[]
    row['grouping'].update(family_id=ident,session_group_id=None,patient_group_id=None,generator_seed_group_id=None)
    row['annotation'].update(primary_intent=label,policy_action='ROUTE',needs_clarification=False,
                             review_status='human_approved',annotators=['software_fixture_A','software_fixture_B'])
    row['provenance']={'software_fixture_only':True,'not_real_human_gold':True}
    return row


def prediction(row,label=None,action='ROUTE'):
    label=label or row['annotation']['primary_intent']
    values={x:.1 for x in LABELS}
    if label:values[label]=.7
    return {'record_id':row['record_id'],'input_hash':hash_inputs(row['inputs']),'prediction':label,
            'probabilities':values if label else None,'action':action,'format_valid':True,'model_run_id':'software_fixture'}


class ContractTests(unittest.TestCase):
    def test_existing_candidates_remain_pending(self):
        rows=read_jsonl(ROOT/'data/candidates.canonical.jsonl')
        self.assertEqual(len(rows),156)
        self.assertTrue(all(r['annotation']['review_status']=='pending' and r['annotation']['primary_intent'] is None for r in rows))

    def test_top_k_and_verbal_confidence_not_full_targets(self):
        for values in [{'triage':.9},{'confidence':.9},{'triage':.5,'preconsult':.5,'info':.5,'other':.5}]:
            with self.assertRaises(ValueError):class_vector(values)

    def test_duplicate_taxonomy_aliases_are_rejected(self):
        with self.assertRaises(ValueError):class_vector({'triage':.5,'GUIDANCE':.5,'preconsult':0,'info':0,'other':0})

    def test_human_gold_requires_independent_review(self):
        row=fixture();row['annotation']['annotators']=['only-one']
        with self.assertRaises(ValueError):validate_record(row)

    def test_risk_cannot_be_routed_or_pseudo_approved(self):
        row=fixture();row['annotation']['risk_trigger']=True
        with self.assertRaises(ValueError):validate_record(row)
        row['annotation']['policy_action']='ESCALATE'
        with self.assertRaises(ValueError):validate_record(row)

    def test_no_synthetic_cross_hospital_results(self):
        row=fixture(partition='test_external');row['grouping']['site_id']='synthetic-site'
        with self.assertRaises(ValueError):validate_record(row)

    def test_literal_evidence_must_match(self):
        row=fixture();row['annotation']['evidence']=[{'field':'user_input','start':0,'end':2,'text':'伪造'}]
        with self.assertRaises(ValueError):validate_record(row)

    def test_test_content_never_enters_synthesis(self):
        row=fixture(partition='test_iid')
        with self.assertRaises(ValueError):assert_teacher_access(row,'synthesize')

    def test_local_patient_data_is_not_implicitly_uploadable(self):
        row=fixture();row['source'].update(kind='hospital_real',rights_status='approved',external_transfer_authorized=False)
        with self.assertRaises(ValueError):assert_teacher_access(row,'label_training')

    def test_family_and_patient_leaks_detected(self):
        a=fixture(ident='a');b=fixture(ident='b',partition='dev_calibration')
        b['grouping']['family_id']=a['grouping']['family_id']
        self.assertTrue(leakage_audit([a,b]))

    def test_augmentation_within_train_is_not_test_leakage(self):
        a=fixture(ident='a');b=fixture(ident='b',partition='train_synthetic')
        b['grouping']['family_id']=a['grouping']['family_id']
        self.assertFalse(leakage_audit([a,b]))

    def test_teacher_soft_labels_not_approved_by_format_alone(self):
        row=copy.deepcopy(read_jsonl(ROOT/'data/candidates.canonical.jsonl')[0])
        row['teacher_outputs'][0]['eligible_for_soft_kd']=True
        with self.assertRaises(ValueError):validate_record(row)


class MetricTests(unittest.TestCase):
    def test_perfect_probabilities_have_zero_proper_scores(self):
        p=[[1 if i==j else 0 for i in range(4)] for j in range(4)]
        result=probability_metrics(LABELS,p)
        self.assertEqual(result['brier'],0);self.assertEqual(result['ece'],0);self.assertEqual(result['aurc'],0)

    def test_confidence_ties_do_not_invent_80_percent_coverage(self):
        result=probability_metrics(LABELS,[[.25]*4]*4)
        self.assertEqual(result['risk_at_coverage'][0]['achieved_coverage'],0)
        self.assertIsNone(result['risk_at_coverage'][0]['risk'])

    def test_dependency_failure_is_an_error_not_other(self):
        rows=[fixture(x,str(i)) for i,x in enumerate(LABELS)]
        preds={r['record_id']:prediction(r) for r in rows};preds['0']={'error':'HTTP_503'}
        result=evaluate(rows,preds)
        self.assertEqual(result['accuracy'],.75);self.assertEqual(result['confusion'][0][4],1)
        self.assertEqual(result['confusion'][0][3],0)

    def test_ambiguity_evaluates_action_without_a_fake_fourth_class(self):
        row=fixture();row['annotation'].update(primary_intent=None,policy_action='CLARIFY',needs_clarification=True)
        validate_record(row)
        result=evaluate([row],{row['record_id']:prediction(row,action='CLARIFY')})
        self.assertEqual(result['n_class_resolvable'],0);self.assertEqual(result['workflow_activation_accuracy'],1)
        self.assertFalse(result['error_counts'])

    def test_missing_clinical_labels_do_not_become_perfect_safety(self):
        row=fixture();result=evaluate([row],{row['record_id']:prediction(row)})
        self.assertIsNone(result['safety_sensitivity'])

    def test_temperature_cannot_be_fitted_on_test(self):
        row=fixture(partition='test_iid')
        with self.assertRaises(ValueError):fit_temperature([row],{row['record_id']:prediction(row)})

    def test_temperature_optimization_does_not_increase_calibration_nll(self):
        rows=[fixture(x,str(i),'dev_calibration') for i,x in enumerate(LABELS)]
        result=fit_temperature(rows,{r['record_id']:prediction(r) for r in rows})
        self.assertLessEqual(result['nll_after'],result['nll_before']+1e-8)

    def test_safety_policy_precedes_confident_intent(self):
        result=decision('info',{'triage':0,'preconsult':0,'info':1,'other':0},True,.9,.2)
        self.assertEqual(result['action'],'ESCALATE')

    def test_policy_thresholds_cannot_be_fitted_on_test(self):
        from care_contract import protocol
        row=fixture(partition='test_iid')
        with self.assertRaises(ValueError):fit_policy([row],{row['record_id']:prediction(row)},1.,protocol()['policy_threshold_fitting'])

    def test_ambiguity_slice_does_not_report_fake_zero_macro_f1(self):
        row=fixture();row['annotation'].update(primary_intent=None,policy_action='CLARIFY',needs_clarification=True)
        result=evaluate([row],{row['record_id']:prediction(row,action='CLARIFY')})
        self.assertIsNone(result['macro_f1'])

    def test_checkpoint_calibration_cannot_cross_to_another_model(self):
        row=fixture();pred=prediction(row);pred['model_run_id']='another-checkpoint'
        cal={'label_order':LABELS,'temperature':1.,'model_run_id':'software_fixture'}
        policy={'selected':{'threshold':.8,'margin':.2},'model_run_id':'software_fixture'}
        with self.assertRaises(ValueError):apply_policy([row],{row['record_id']:pred},cal,policy)

    def test_missing_output_survives_policy_wrapper_as_an_error(self):
        row=fixture();cal={'label_order':LABELS,'temperature':1.,'model_run_id':'software_fixture'}
        policy={'selected':{'threshold':.8,'margin':.2},'model_run_id':'software_fixture'}
        values=apply_policy([row],{},cal,policy)
        self.assertEqual(values[0]['input_hash'],hash_inputs(row['inputs']))
        self.assertEqual(evaluate([row],{row['record_id']:values[0]})['accuracy'],0)


class SelectionAndStatisticsTests(unittest.TestCase):
    def test_acquisition_cannot_see_test_or_calibration(self):
        with self.assertRaises(ValueError):select([fixture(partition='test_iid')],1,'random')

    def test_random_selection_is_reproducible_and_budgeted(self):
        rows=[fixture(ident=str(i),partition='train_unlabeled') for i in range(10)]
        self.assertEqual(select(rows,3,'random'),select(rows,3,'random'))
        self.assertEqual(len({x['record_id'] for x in select(rows,3,'random')}),3)

    def test_holm_is_step_down_and_monotone(self):
        adjusted=holm({'a':.01,'b':.03,'c':.2})
        self.assertAlmostEqual(adjusted['a'],.03);self.assertAlmostEqual(adjusted['b'],.06);self.assertAlmostEqual(adjusted['c'],.2)

    def test_identical_models_have_zero_paired_effect(self):
        rows=[fixture(x,str(i)) for i,x in enumerate(LABELS)]
        preds={r['record_id']:prediction(r) for r in rows}
        result=paired_comparison(rows,[preds]*5,[preds]*5,[11,23,42,67,101],replicates=50)
        self.assertEqual(result['delta'],0);self.assertEqual(result['delta_ci95'],[0,0]);self.assertEqual(result['cluster_permutation_p'],1)

    def test_missing_comparison_outputs_are_not_dropped(self):
        rows=[fixture(x,str(i)) for i,x in enumerate(LABELS)]
        preds={r['record_id']:prediction(r) for r in rows}
        result=paired_comparison(rows,[{}],[preds],[42],replicates=20)
        self.assertEqual(result['left_mean'],0);self.assertEqual(result['right_mean'],1)
        self.assertEqual(result['left_missing_outputs'],[4])

    def test_safety_200_total_does_not_prove_sub_one_percent(self):
        result=safety_sample_size_report()
        self.assertGreater(result['zero_miss_upper_rates']['100'],.01)
        self.assertEqual(result['positive_cases_required_with_zero_misses']['0.01'],299)


class AnnotationAndPolicyTests(unittest.TestCase):
    def test_blind_packets_do_not_expose_teacher_or_proposed_labels(self):
        packet=read_jsonl(ROOT/'data/annotation_you.v1.jsonl')[0]
        self.assertNotIn('teacher_outputs',packet);self.assertNotIn('provenance',packet)
        self.assertIsNone(packet['annotation']['primary_intent']);self.assertTrue(packet['family_id'].startswith('family-'))

    def test_blank_agreement_does_not_become_gold(self):
        from merge_annotations import merge
        base=fixture();base['annotation'].update(primary_intent=None,policy_action=None,needs_clarification=None,review_status='pending',annotators=[])
        packet={'record_id':base['record_id'],'policy_version':base['policy_version'],'inputs':base['inputs'],'input_hash':hash_inputs(base['inputs']),'annotation':copy.deepcopy(base['annotation'])}
        accepted,queue,stats=merge([base],[packet],[copy.deepcopy(packet)])
        self.assertFalse(accepted);self.assertEqual(len(queue),1)

    def test_ordinary_dual_review_can_produce_gold(self):
        from merge_annotations import merge
        base=fixture();packet={'record_id':base['record_id'],'policy_version':base['policy_version'],'inputs':base['inputs'],'input_hash':hash_inputs(base['inputs']),'annotation':copy.deepcopy(base['annotation'])}
        base['annotation'].update(primary_intent=None,review_status='pending',annotators=[])
        accepted,queue,stats=merge([base],[packet],[copy.deepcopy(packet)])
        self.assertEqual(len(accepted),1);self.assertFalse(queue)

    def test_nonclinical_dual_review_cannot_approve_escalation(self):
        from merge_annotations import merge
        base=fixture();base['annotation'].update(policy_action='ESCALATE',risk_trigger=True)
        packet={'record_id':base['record_id'],'policy_version':base['policy_version'],'inputs':base['inputs'],'input_hash':hash_inputs(base['inputs']),'annotation':copy.deepcopy(base['annotation'])}
        accepted,queue,stats=merge([base],[packet],[copy.deepcopy(packet)])
        self.assertFalse(accepted);self.assertEqual(len(queue),1)

    def test_policy_search_does_not_use_gold_risk_as_inference_flag(self):
        from care_contract import protocol
        row=fixture(partition='dev_policy');row['annotation'].update(policy_action='ESCALATE',risk_trigger=True,risk_review_status='clinician_approved')
        pred=prediction(row);pred['independent_safety_flag']=False
        result=fit_policy([row],{row['record_id']:pred},1.,protocol()['policy_threshold_fitting'])
        self.assertTrue(all(x['mean_research_cost']==50 for x in result['grid']))


if __name__=='__main__':unittest.main()
