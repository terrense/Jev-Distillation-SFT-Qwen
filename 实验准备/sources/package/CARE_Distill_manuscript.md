# CARE-Distill: Confidence-Aware and Ambiguity-Resolved Distillation for Safe Clinical Intent Routing

**Research manuscript — pre-experimental draft | 18 November 2026**  
**Authors:** [To be specified by the research team]  
**Status:** Methods and experimental protocol complete; all numerical results and empirical claims require actual experiments.  
**Target domains:** clinical-facing NLP, knowledge distillation, dialogue systems, human-centered AI.

## Abstract

Intent routing is a low-latency but consequential component of hospital-facing conversational agents. A deployed router must distinguish department navigation, pre-consultation information gathering, hospital-service inquiries, and out-of-scope requests, while handling under-specified utterances, overlapping intents, multi-turn context, and safety-critical situations. Although stronger general-purpose language models can provide inexpensive weak supervision at scale, directly fine-tuning a small model on their hard labels can amplify annotation errors, obscure class ambiguity, and encourage overconfident routing. We propose **CARE-Distill**, a research framework that combines (i) a four-way primary intent taxonomy with independent clarification and safety actions, (ii) heterogeneous teacher supervision from a generative model and an optional typed-decision model, (iii) disagreement-aware annotation and student-guided acquisition of difficult instances, and (iv) confidence-calibrated selective routing. The design supports black-box teachers through hard-label distillation and, when valid class distributions are accessible, an additional class-level soft-target objective. We specify reproducible comparisons against encoder classifiers, prompted teachers, Qwen3 students, standard supervised fine-tuning, synthetic-data distillation, and student-driven correction. Evaluation is planned on expert-adjudicated, session-disjoint, difficult-case, temporal, and cross-hospital splits, with multi-seed stability, calibration, risk, latency, and cost analyses. **No outcome values are reported in this pre-experimental manuscript**; the central hypothesis is that targeted, calibrated teacher supervision improves routing reliability at a fixed annotation and inference budget, especially on ambiguous clinical requests.

**Keywords:** knowledge distillation; intent classification; clinical dialogue; selective prediction; active learning; Qwen3; multi-teacher supervision; ambiguity; calibration.

## 1 Introduction

Hospital conversational systems increasingly separate language understanding from downstream agents responsible for department navigation, pre-consultation questioning, administrative service assistance, and other requests. In such an architecture, the first-stage router determines which workflow is activated. A wrong route may produce irrelevant follow-up questions, unnecessary tool calls, or delayed escalation. Unlike conventional intent classification, hospital requests regularly combine clinical content and institutional operations. “I booked cardiology, but I have chest pain now—should I change departments?” mixes a service request, a clinical symptom, and a possible urgent-safety cue. Thus, high aggregate accuracy alone does not establish reliable routing.

A powerful generative teacher can label intent data and synthesize instructive examples, reducing the need for exhaustive manual annotation. Yet large teacher models may disagree on close boundaries and often supply apparently confident but weakly justified labels. Recent studies of intent understanding, disambiguated synthesis, and multi-turn labeling identify ambiguity and intent switching as central challenges [3–6]. Studies of clinical weak supervision show that teacher-generated labels can be useful while remaining inferior to carefully curated human supervision in some regimes [14]. Moreover, a recent healthcare function-routing benchmark found that knowledge distillation outcomes for small students could vary dramatically across random seeds, including collapse and output truncation [13].

We investigate an operationally narrow but practically important question: **Can a small, locally deployable model learn reliable clinical intent routing from a small expert-labeled core and selectively acquired, heterogeneous teacher supervision—without sacrificing calibrated uncertainty or cross-site generalization?** We deliberately avoid treating ordinary four-class supervised fine-tuning as a new algorithm. Our proposed contribution is the *combined decision-and-supervision protocol* for ambiguity-aware, risk-sensitive clinical routing, together with a rigorous evaluation of whether each component is actually necessary.

The study will make four testable contributions if the proposed experiments support them. First, we specify a hierarchical output schema separating the primary workflow label from ambiguity, secondary intents, clarification, and safety escalation. Second, we define a teacher adjudication and sample-selection pipeline that distinguishes reliable pseudo-labels from contested examples requiring human review. Third, we propose a student model with class-level distillation and calibrated selective decisions while retaining a low-cost encoder alternative. Fourth, we provide a multi-hospital and multi-seed evaluation protocol that explicitly measures unsafe routing, over-escalation, and latency/cost trade-offs rather than accuracy alone.

[FIGURE 1 PLACEHOLDER — Hospital intent-routing architecture: ASR/text → conversation state → compact student router → route / clarify / safety escalation → downstream clinical or service agent, with offline teacher loop.]

## 2 Related Work

### 2.1 Large-to-small distillation and teacher formats

Classical knowledge distillation transfers probabilistic predictions rather than only one-hot labels [1]. In the LLM setting, teacher-generated task examples, explanations, and synthetic queries can serve as supervision even when complete logits are unavailable [2,9]. Recent work on calibrated progressive distillation and score matching emphasizes that teacher–student capacity gaps and distribution-matching objectives influence knowledge transfer [16,20]. However, label classification differs from free-form generation: a compact encoder may be a stronger efficiency baseline than an 8B autoregressive model.

### 2.2 Intent classification, clarification, and data synthesis

Nandi et al. [3] jointly study lightweight intent classification and clarifying-question generation. Castillo-López et al. [4] show that class-conditioned synthetic utterances can unintentionally overlap with other intent categories. Xie et al. [5] address multi-turn intent labeling and intent changes; Castillo-López et al. [23] explore turn-taking dynamics as an additional representation of conversational state. Wulamu et al. [6] use synthetic, adjudicated, multi-intent data for spoken language understanding; Sun and Sang [7] emphasize retrieval, reranking, and selective escalation as strong non-fine-tuning alternatives. Consequently, a four-class model trained on synthetic paraphrases is not sufficient as a research contribution unless evaluated against these techniques.

### 2.3 Active learning, typed decisions, and heterogeneous teachers

ActiveLLM [8] addresses low-resource example selection. Jev is a commercial typed-decision model evaluated for classification and routing [10]; it is not an unrestricted text generator. Sadoun et al. [11] explore combining Jev with a generative teacher, while Wu et al. [12] study a student-guided teacher pipeline for fixed-category task routing. These studies motivate comparing Jev as a stand-alone baseline and as an annotation source; their results cannot be assumed to transfer to hospital data. When a teacher exposes only choices or a single confidence rather than a normalized four-class distribution, we must not treat it as a full soft-label oracle.

### 2.4 Clinical classification and robustness

Weng et al. [14] compare human and LLM-generated annotations in multisite radiology negation detection. Zhang [15] shows that apparently high aggregate clinical pragmatic recognition can conceal difficult-case gaps. Górski and Czyżewski [21] describe teacher-to-encoder distillation for medical text tagging. Sumit et al. [13] show why multiple seeds and function-format failure analysis are particularly important for healthcare routing. While guided on-policy distillation and student-centered trajectory correction [17–19] are powerful for multi-turn agents, a static four-class classifier should not be described as token-level OPD merely because errors are relabeled. We reserve the term *student-guided hard-example acquisition* for our primary method and treat genuine on-policy, token-level distillation as an optional, separately controlled extension.

## 3 Problem Definition and Annotation Ontology

### 3.1 Inputs and outputs

For conversation episode e at decision turn t, let x=(u_t,h_{t-1},c_t), where u_t is the latest utterance, h_{t-1} is bounded dialogue history, and c_t includes permitted workflow state (e.g., whether the system is currently collecting mandatory pre-consultation slots). The primary intent y belongs to {GUIDANCE, PRECONSULT, SERVICE, OTHER}. The auxiliary outputs are a set M of secondary intents, an ambiguity indicator a, a safety trigger z, and a next-system action d in {ROUTE, CLARIFY, ESCALATE}. The primary label describes the user's dominant *requested workflow*, not a diagnosis or disease category.

### 3.2 Primary taxonomy

| Label | Operational meaning | Positive example | Important boundary |
|---|---|---|---|
| GUIDANCE / 导诊 | User requests department or care-path navigation | “经常头晕应该挂哪个科？” | Asking which department is guidance; collecting mandatory history after registration is pre-consultation. |
| PRECONSULT / 预问诊 | User is supplying or is asked for clinical history before clinician interaction | “症状有三天了，昨晚开始加重。” in pre-consultation context | A symptom statement alone may be ambiguous without dialogue state. |
| SERVICE / 流程信息咨询 | Appointment, payment, schedule, facility or hospital-process information | “MRI检查怎么预约，收费在哪里看？” | Department selection versus booking procedure must be differentiated. |
| OTHER / 其他 | Outside the three supported workflows, including greetings and unrelated tasks | “帮我写一首诗。” | Emergency danger is **not** simply routed to OTHER; the safety trigger is independent. |

### 3.3 Ambiguity and multi-intent rules

A single utterance may support multiple actions: “我挂了心内科，想改成急诊，要在哪里操作？” contains both routing and service intents. Annotators assign the primary intent according to the user's immediate requested outcome under a written hierarchy: explicit action goal; current dialogue stage; preceding question; finally a CLARIFY action if context remains insufficient. Secondary intent labels preserve overlap rather than forcing spurious certainty. The CLARIFY label is an *action*, not a fifth top-level intent; an uncertain example may still have a distribution over the four primary categories. For urgent-risk content, a separately defined rule-and-review safety layer can force ESCALATE independently of the classifier.

### 3.4 Medical safety boundary

The router is a workflow selector, not a diagnostic or triage-grade medical device. It must not claim to diagnose chest pain, determine urgency from text alone, or suppress escalation because a task looks administrative. A clinician-approved set of critical patterns (e.g., loss of consciousness, severe breathing difficulty) flags for specialized review and escalation. Clinicians must define escalation policy, failure severity, and adjudication standards before model selection. False-negative safety escalation, false-positive escalation, and delayed workflow resolution are measured separately.

## 4 Proposed CARE-Distill Framework

### 4.1 System overview

The data flow is: consented/de-identified task logs → session-level deduplication and label-ontology validation → expert gold subset → heterogeneous teacher annotation of the unlabeled training pool → agreement/disagreement audit and synthetic hard-case generation → small-student training → student-guided difficult-example acquisition → calibrated route/clarify/escalate policy → prospective-style shadow evaluation. Test splits remain inaccessible to teachers, human-in-the-loop data selection, hyperparameter tuning, and prompt refinement.

[FIGURE 2 PLACEHOLDER — CARE-Distill training diagram with teacher ensemble, adjudication gates, student hard-case mining, and calibration head.]

### 4.2 Teacher layer: generative and typed-decision supervision

Three teacher conditions are specified: (T1) a version-locked, authorized high-capability GPT-class model that returns one JSON decision, concise observable evidence, and an ambiguity flag; (T2) an optional Jev-like typed-decision API that accepts the four permitted outcomes and returns its supported choice/probability interface; and (T3) a self-hosted Qwen3-32B or Qwen3-30B-A3B teacher. Availability and licensing govern the final teacher choices. Text-only labels and verbalized confidence are **not** equivalent to token logits or a normalized posterior. If a service does not expose a complete probability vector, we use hard-label targets, independently validated repeated-choice estimates where permitted, or human adjudication rather than fabricate soft probabilities.

For each candidate, record teacher version, full input context, prompt and schema versions, class prediction, permitted confidence representation, parse status, ambiguity flag, evidence span (a literal span of the input, not a free-form medical rationale), and latency/cost. Teacher agreement is a screening signal, not proof of correctness. All uncertain high-risk examples require human review.

### 4.3 Disagreement-aware supervision

Let H be expert-labeled data and U the unlabeled candidate pool. For each x in U, obtain predictions from the available teachers. Estimate teacher reliability only on a nested calibration fold within H, never on the held-out test. Label disagreement, low student margins, close-boundary semantic similarity, and risk indicators define a review queue. A teacher pseudo-label is eligible for automatic training only when its schema is valid, its task mapping is unambiguous under the annotation manual, and it passes a preregistered agreement/reliability rule. Disputed samples are not discarded wholesale: a separate ambiguity pool is expert-adjudicated and used for CLARIFY supervision.

If a full four-class teacher distribution q_T is legitimately available, it may supervise the Student’s four-class probabilities p_θ. Otherwise, for teacher j, use its verified hard label y_j and a reliability weight w_j. For multiple complete posteriors, the exact aggregation rule (e.g., calibrated weighted mean of distributions) is fixed before testing. Calibration must precede aggregation, and probabilities are not inferred from arbitrary verbal confidence scales.

### 4.4 Student-guided difficult-example acquisition

After initial supervised training, apply the student to an unlabeled training-only pool. Candidate priority is a preregistered combination of (a) low student class margin, (b) teacher–student disagreement, (c) embedding-space diversity relative to already selected examples, and (d) clinical workflow impact or safety relevance. Selection is budgeted: compare selecting N=250, 500, 1,000, 2,000 new examples per round with a random-annotation control. Critical safety cases are always escalated to clinician adjudication; teacher-only agreement cannot authorize a safety label. Each round retrains from a documented checkpoint under matched compute, avoiding hidden increases in training budget.

This is **student-driven annotation** rather than standard token-level OPD. As an optional experiment, a self-hosted open-weight teacher may score the student’s own generated multi-turn routing decisions to perform genuine on-policy KL training; this experiment is only valid when both models’ token/probability interfaces, rollouts, and loss masking are explicitly implemented and reported.

### 4.5 Multi-task objectives and class-level distillation

Let p_θ(y|x) be the student’s four-class probability and a_θ(d|x) its action distribution. Gold-label training uses L_gold = CE(y,p_θ). If class-level q_T is available, let L_soft = KL(q_T^τ || p_θ^τ), where τ is the recorded calibration/distillation temperature. A pseudo-label loss L_pseudo uses CE on audited training-only labels with reliability weights. Action prediction uses L_action = CE(d,a_θ), and safety modeling uses L_safety = BCE(z, z_θ), where safety targets are human-approved. A consistency term L_cons can penalize inconsistent predictions on verified intent-preserving paraphrase pairs, with the caveat that many apparent paraphrases alter clinical meaning and must be filtered.

The candidate optimization objective is:

**L_total = L_gold + λ_p L_pseudo + λ_s L_soft + λ_a L_action + λ_z L_safety + λ_c L_cons.**

Only terms supported by actual supervision are active. λ values are tuned on development data; none is declared superior before ablation. For generative Qwen3 students, the preferred primary experiment uses a constrained four-label output or a dedicated classification head rather than unrestricted long CoT. For encoder baselines, use an ordinary four-class head under matched label budgets.

### 4.6 Calibrated selective inference

On the development split, fit temperature scaling or an alternative prespecified calibration method to the four-class student probabilities. At inference, if the independent safety rule fires, return ESCALATE; otherwise route when calibrated confidence and class margin exceed thresholds; otherwise emit CLARIFY (or a supervised fallback when clarification is impossible). Thresholds optimize a predeclared cost-sensitive utility and are locked before the test evaluation. Calibration reduces some confidence errors but does not guarantee correctness under distribution shift. Every branch is auditable through reason codes, not unrestricted medical explanations.

[FIGURE 3 PLACEHOLDER — Risk–coverage curves for selective routing and a calibration/reliability plot on IID, hard, and held-out hospital splits.]

## 5 Data and Study Design

### 5.1 Data sources and ethics

The study is planned around authorized historical hospital-assistant conversation logs, including de-identified records of department navigation, pre-consultation collection, appointment/fee/process assistance, and unrelated requests. Any use of patient or hospital data requires institutional authorization, appropriate privacy/legal review, removal or minimization of protected personal information, and restrictions on transfer to external APIs. De-identification does not by itself authorize uploading conversations to commercial teachers. Until authorization exists, develop with wholly synthetic/public data and self-hosted teachers. No proprietary real hospital data or model scores are claimed to have been obtained for this draft.

### 5.2 Proposed data budget (planning values, not observed counts)

The planned working pool contains 24,000 non-duplicated, authorized real dialogue decisions. Reserve 6,000 expert-gold instances for training/development/evaluation and 18,000 strictly training-side unlabeled decisions. Pseudo-label up to 12,000 of the latter after quality gates, then optionally generate 10,000 synthetic contrastive or paraphrastic examples. These are **design targets** subject to data availability, quality, and ethics approval, not reported dataset statistics.

| Human-adjudicated split | Planned N | Use | Isolation rule |
|---|---:|---|---|
| Gold train | 2,000 | Pure human-supervision baseline and teacher calibration subfold | Training hospitals; session disjoint |
| Development | 600 | Hyperparameter, prompt and threshold selection | Disjoint sessions, training hospitals |
| IID test | 1,200 | Primary expert evaluation | Never used in tuning |
| Hard-boundary test | 800 | Ambiguous, overlapping, ASR-noisy, missing-context items | Hand-curated, test only |
| External-hospital test | 800 | Unseen site evaluation | Entire hospital held out |
| Temporal test | 400 | Future-time drift | Later collection period |
| Safety challenge test | 200 | Critical escalation and unsafe-route audit | Clinician verified, test only |
| **Total** | **6,000** | **Planned gold count** | **No conversation may straddle splits** |

Multi-turn episodes and ambiguity/safety slices must be separately tagged within these partitions; the same item cannot be claimed as independent in two test denominators. Session, patient, and site pseudonyms are used only under permitted governance; near duplicates are detected by lexical and embedding matching. For the external-site test, data from that site are excluded from teacher few-shot prompts, student training, and calibration.

### 5.3 Annotation protocol

Two annotators independently label each gold item using a versioned bilingual guideline. A clinician or trained clinical operations reviewer adjudicates cases involving clinical urgency or an unresolved workflow boundary. Record primary/secondary labels, action, risk flag, evidence from the utterance, full context visibility, and whether the instance was confidently resolvable. Report raw agreement, Cohen’s kappa for primary labels when applicable, and Krippendorff’s alpha or appropriate multi-label agreement for auxiliary decisions. Publish de-identified annotation templates and aggregate label statistics, not unauthorized patient utterances.

### 5.4 Contrastive and difficult-case construction

Construct minimal pairs that deliberately vary the requested workflow while preserving surface vocabulary: “应该挂什么科？” versus “已经挂号，怎么改科室？”; “我头晕三天了” within a pre-consultation slot-filling turn versus the same utterance in a new unscoped conversation. Other challenge categories include dialect/ASR substitutions, incomplete requests, intent switching, multiple simultaneous goals, irrelevant context, policy updates, hospital-specific synonyms, and risk-bearing language inside administrative requests. Since synthetic augmentation can introduce unintended class ambiguity [4,22], a verified ambiguity may be retained as a CLARIFY example rather than force-labeled to the generation target.

## 6 Experimental Protocol

### 6.1 Research questions

**RQ1 (Data efficiency).** How much does teacher-generated training data improve performance beyond the same 2,000 expert labels at 100, 250, 500, 1,000, and 2,000 human-label budgets?

**RQ2 (Teacher selection).** Do GPT-class, Jev-type, local Qwen teachers, and their consensus differ in pseudo-label quality and downstream student quality when model/API cost is controlled?

**RQ3 (Training method).** Which components contribute independently: raw synthetic data, ambiguity filtering, supervised hard-label distillation, class-soft-label distillation, student-guided selection, and calibrated abstention?

**RQ4 (Robustness).** Do any gains persist on multi-turn, hard-boundary, cross-hospital, temporal, ASR-noisy, and safety subsets?

**RQ5 (Cost and stability).** Is Qwen3-8B justified over compact encoder classifiers or smaller Qwen3 students, in accuracy, calibration, multi-seed variance, latency, and recurring inference cost?

### 6.2 Baselines

| ID | Baseline | Supervision / inference configuration |
|---|---|---|
| B0 | Majority + lexical rules | Fixed rules, no teacher |
| B1 | TF-IDF + linear SVM / logistic regression | Human-gold only |
| B2 | Chinese MacBERT / modern Chinese encoder | Human-gold classification head |
| B3 | BGE embedding + linear classifier | Human-gold only; frozen encoder |
| B4 | Qwen3-0.6B / 1.7B / 4B / 8B prompted | Zero/few-shot, constrained labels |
| B5 | Qwen3-8B supervised SFT | Human-gold only; matched train budget |
| B6 | Teacher pseudo-label SFT | Hard labels, no extra modules |
| B7 | Teacher + rationale/brief evidence SFT | Same pseudo-label count and context |
| B8 | Disambiguated synthetic augmentation | DDAIR-like ambiguity screening |
| B9 | Selective capability routing | Small model + budgeted GPT/Jev fallback |
| B10 | CARE-Distill full candidate | Heterogeneous teachers + acquired examples + calibrated actions |

External Jev is evaluated as a direct zero-shot typed decision baseline if authorized access exists. We also evaluate a self-hosted strong Qwen3 teacher. To avoid unfair comparisons, human-label counts, selection budgets, teacher calls, student architecture, maximum context, training steps, and allowed external API spend are explicitly reported.

### 6.3 Proposed student and optimization settings

Primary student: Qwen3-8B with a 4-class head plus optional action and safety heads or tightly constrained structured output. Secondary students: Qwen3-1.7B and Qwen3-4B if available. Critical comparison: a tuned encoder-only classifier. Candidate Qwen LoRA search: rank {16,32}, scaling {32,64}, learning rate {5e-5,1e-4,2e-4}, 2–4 epochs, effective batch size {32,64}, context 256 tokens single-turn / 1024 tokens multi-turn; calibrate final choices on development only. Encoder search: learning rate {1e-5,2e-5,3e-5}, 3–8 epochs. Fixed validation triggers early stopping. All settings are hypotheses; selected configurations and computation are disclosed per seed. Use a common tokenizer and constrained output grammar for generation baselines. Report actual GPU type, peak VRAM, tokens/s, exact hardware software versions, and teacher request costs; no cost is assumed in advance.

### 6.4 Sampling and teacher-budget study

At each budget level (250, 500, 1,000, 2,000), compare random selection, margin uncertainty, disagreement-only, diversity-only, risk-aware but uniformly adjudicated, and the full combined selector. A student-acquisition cycle cannot inspect test samples and cannot change teacher prompts after seeing test outcomes. The same total number of teacher judgments and clinician adjudications must be used for each budget comparison.

### 6.5 Ablation matrix

| Variant | Multi-teacher | Ambiguity pool | Student-guided acquisition | Soft class targets | Calibrated abstention |
|---|---|---|---|---|---|
| A0 Gold-only SFT | No | No | No | No | No |
| A1 + Teacher hard labels | Single | No | No | No | No |
| A2 + Multi-teacher audited labels | Yes | No | No | No | No |
| A3 + Boundary examples | Yes | Yes | No | No | No |
| A4 + Active selection | Yes | Yes | Yes | No | No |
| A5 + Soft targets (when valid) | Yes | Yes | Yes | Yes | No |
| A6 Full CARE-Distill | Yes | Yes | Yes | Valid teachers only | Yes |
| A7 Remove safety head | Yes | Yes | Yes | As in A6 | Yes, no auxiliary safety |
| A8 Remove clinical context | Yes | Yes | Yes | As in A6 | Yes |
| A9 Optional genuine OPD | Open-weight teacher | Yes | On-policy rollouts | Token level | Yes |

The A9 experiment is *not* identified with ordinary student-guided hard-case mining. If it cannot be run end-to-end, it is omitted rather than reported as a completed OPD ablation.

### 6.6 Metrics and statistical analysis

Primary: expert-gold macro-F1 for four-class intent prediction on IID and hard-boundary tests. Secondary: balanced accuracy, per-class F1, secondary-intent exact/micro-F1 when available, confusion matrix, CLARIFY precision/recall, route accuracy conditional on action, safety-escalation sensitivity and specificity, and over-escalation burden. Calibration: Brier score, ECE with a fixed binning rule, negative log likelihood, AURC, and selective risk at target coverage 80%, 90%, and 95%. Practical systems: p50/p95 latency, throughput, peak GPU memory, token cost, annotation cost, escalation fraction, and end-to-end correct downstream workflow activation.

For key models, run at least five independent seeds; report mean ± standard deviation and per-seed raw values. Use patient/session-clustered paired bootstrap (10,000 replicates) for 95% confidence intervals where sampling permits, exact or McNemar-style paired testing for comparable discrete predictions, and a prespecified multiplicity correction (Holm) across primary contrasts. Site-level intervals and worst-site performance must be reported. Do not declare an effect from a single favorable seed or unpaired train/test changes. Because recent routing-distillation results demonstrate severe seed instability [13], stability is treated as a first-class endpoint.

### 6.7 Failure taxonomy and manual review

Analyze at least: GUIDANCE↔PRECONSULT confusion; GUIDANCE↔SERVICE confusion; OTHER over-prediction; missing dialogue-state context; multi-intent dominance errors; premature route vs clarification; bad escalation; ASR corruption; teacher label hallucination; student format invalidity; and brittle hospital-specific terms. For each group, publish counts, examples only when authorized, root-cause adjudications, and whether the error came from labels, sample selection, optimization, or decision thresholds. The explanation/rationale generated by a teacher is not treated as independently validated clinical evidence.

[FIGURE 4 PLACEHOLDER — Multi-seed boxplot of macro-F1 and seed-collapse indicators by distillation method.]
[FIGURE 5 PLACEHOLDER — Cost–quality Pareto frontier across encoder, Qwen3-1.7B/4B/8B, Jev/GPT teacher, and hybrid fallback.]

## 7 Results (to be completed only after experiments)

### 7.1 Main performance

[TABLE 5 PLACEHOLDER — IID macro-F1, hard macro-F1, external-site macro-F1, safety escalation recall, p95 latency and GPU memory; include mean ± SD across seeds and confidence intervals.]

**Required result narrative:** Report whether the full method improves the primary metrics over (1) gold-only encoder, (2) gold-only Qwen3-8B, (3) simple teacher-label SFT, and (4) a selective fallback system. State negative findings if the simple baseline equals or outperforms CARE-Distill. Only describe statistically supported differences as meaningful.

### 7.2 Ablations and teacher comparisons

[TABLE 6 PLACEHOLDER — Teacher single/multi consensus; individual modules; quality-versus-labeling budget; compare hard and soft teacher targets only where distributions are valid.]

### 7.3 Reliability, safety, and distribution shift

[TABLE 7 PLACEHOLDER — Selective risk/coverage, false escalations, missed escalations, calibration and temporal/site shifts.]

### 7.4 Qualitative and cost analysis

[TABLE 8 PLACEHOLDER — Audited false-route cases; latency/cost breakdown and seed stability.]

No efficacy, deployment, or publication claim is justified before these tables have actual data. If gains disappear under patient-level or site-level separation, the paper should present that result as a limitation and potentially reposition the contribution as a rigorous clinical routing benchmark rather than a new superior algorithm.

## 8 Discussion

The first unresolved issue is model choice. Four-class classification is not inherently a task requiring an 8B autoregressive decoder; an encoder may meet the accuracy, latency, and privacy targets with far less compute. If Qwen3-8B is useful only when performing richer conversation-state interpretation and issuing clarifying questions, that advantage must be isolated from model size and from the downstream agent. A second issue is teacher reliability. A probabilistic typed-decision model and a generative model may fail differently, but teacher agreement can still reflect shared dataset bias, incorrect taxonomy boundaries, or the same linguistic shortcut. Third, synthetic boundary cases may improve one test slice while harming real naturalistic samples; human adjudication and strict cross-source isolation are necessary. Fourth, external-site and future-time evaluations are particularly informative because hospital names, department taxonomies, and scheduling terminology evolve.

An attractive extension is to train a small encoder for the four-way decision and reserve Qwen3-8B for genuinely uncertain, multi-turn, or clarification-dependent cases. A well-calibrated cascade could be operationally better than forcing every message through an 8B model. This is not a retreat from distillation: the central research question concerns the best allocation of teacher supervision, human review, and runtime compute, not allegiance to a particular student architecture.

## 9 Limitations, Ethics, and Reproducibility

This is a proposed research study, not an evaluated clinical system. Limitations include possible teacher and annotator biases, different definitions of “primary intent” across institutions, public/private data access constraints, closed-model version drift, uncertainty in API probability semantics, and changing cost/latency environments. Safety escalation requires independent clinical and institutional authorization; the system must not offer autonomous diagnosis or prevent human access. All private logs must be accessed under explicit institutional approval. Retain audit metadata only to the extent permitted, follow data retention controls, and avoid releasing protected patient data or prompting external vendors with unauthorized text. Document dataset provenance, annotation protocols, random seeds, prompt versions, model checkpoint hashes, training scripts, compute, and the protocol for any prospective shadow evaluation.

## 10 Conclusion

We present a complete *proposed* framework for distilling strong teacher supervision into locally deployable hospital-intent routers. CARE-Distill separates primary intent from clarification and safety, uses typed and generative teachers without assuming unsupported logit access, prioritizes student-relevant difficult examples, and adds calibrated selective routing. Its contribution will depend on rigorous evidence: robust gains over compact encoder baselines, low variance across seeds, better performance on ambiguous multi-turn examples, and safe, efficient operation across hospitals. These claims remain hypotheses until the prescribed experiments are performed.

## References

[1] Hinton, G., Vinyals, O., and Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531. https://arxiv.org/abs/1503.02531

[2] Hsieh, C.-Y. et al. (2023). Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes. arXiv:2305.02301. https://arxiv.org/abs/2305.02301

[3] Nandi, S., Agarwal, T., Singh, A., and Bhatt, P. (2026). Scaling Intent Understanding: A Framework for Classification with Clarification using Lightweight LLMs. EACL 2026 Industry Track, 181–192. https://aclanthology.org/2026.eacl-industry.14/

[4] Castillo-López, G., Lombard, A., Semmar, N., and de Chalendar, G. (2026). How DDAIR you? Disambiguated Data Augmentation for Intent Recognition. EACL 2026 Short Papers, 274–286. https://aclanthology.org/2026.eacl-short.20/

[5] Xie, B., Bian, K., Gusain, J. R. S., Ramanathan, M., and Maragoud, R. (2026). Building Multi-turn Intent Classification with LLM-based Labeling. CustomNLP4U 2026, 68–83. https://aclanthology.org/2026.customnlp4u-1.8/

[6] Wulamu, A., He, A., Zhang, D., Li, J., Qian, X., Zhu, Z., Qin, L., and Xing, B. (2026). Bridging Generation and Parsing: Collaborative Self-Distillation for Zero-shot Multi-Intent Spoken Language Understanding. EMNLP 2026, 15850–15866. https://aclanthology.org/2026.emnlp-main.689/

[7] Sun, J., and Sang, Y. (2026). Cost-Efficient Intent Classification via Semantic-Candidate Adaptation and Selective Capability Routing. EMNLP 2026 Industry Track. https://aclanthology.org/2026.emnlp-industry.257/

[8] Bayer, M., Lutz, J., and Reuter, C. (2026). ActiveLLM: Large Language Model-Based Active Learning for Textual Few-Shot Scenarios. Transactions of the ACL, 14:1–22. https://aclanthology.org/2026.tacl-1.1/

[9] Pecher, B., Cegin, J., Belanec, R., Srba, I., Simko, J., and Bielikova, M. (2026). Better as Generators Than Classifiers: Leveraging LLMs and Synthetic Data for Low-Resource Multilingual Classification. arXiv:2601.16278. https://arxiv.org/abs/2601.16278

[10] Deußer, T., Sparrenberg, L., and Sifa, R. (2026). Evaluating and Benchmarking the System One Model Jev. arXiv:2609.37647. https://arxiv.org/abs/2609.37647

[11] Sadoun, M. Y., Sharif, S., and Banad, Y. M. (2026). From Probabilities to Decisions: Search and Multi-Teacher Distillation with Jev. arXiv:2610.09188. https://arxiv.org/abs/2610.09188

[12] Wu, H., Manoharan, S., Wan, J., Tu, F., Zhao, J., and Chen, X. (2026). Student-Guided Teacher Distillation for Efficient LLM Task Routing: Positioning Against Jev-Style System-1 Classifiers. arXiv:2610.02516. https://arxiv.org/abs/2610.02516

[13] Sumit, D., Haque, S. U., and Sadeque, F. (2026). Below the Noise Floor: Bimodal Seed Collapse and Distinct Failure Modes in Small-Model Knowledge Distillation. Findings of EMNLP 2026, 43917–43928. https://aclanthology.org/2026.findings-emnlp.2063/

[14] Weng, K.-H., Huang, B.-L., Kuo, Y.-T., Chou, Y.-C., Hsieh, T.-J., and Liu, C.-F. (2026). Reducing Annotation Costs Using Knowledge Distillation for Label-Efficient Negation Detection in Radiology Reports: Multisite Internal Validation Study. JMIR Medical Informatics, 14, e103912. https://medinform.jmir.org/2026/1/e103912/

[15] Zhang, K. (2026). PCP: A Controlled Benchmark for Pragmatic Intention Recognition in Clinical Communication. LUHME 2026, 1–24. https://aclanthology.org/2026.luhme-1.1/

[16] Zhang, M., and Liu, L. (2026). Calibrated Progressive Distillation: Co-Designing Curriculum and Target Mixing for Knowledge Distillation of Large Language Models. Findings of ACL 2026, 6757–6776. https://aclanthology.org/2026.findings-acl.335/

[17] Jin, W., Min, T., Yang, Y., Wei, D., Zhou, Y., Kadhe, S. R., Baracaldo, N., and Lee, K. (2026). Entropy-Aware On-Policy Distillation of Language Models. ICML 2026, Proceedings of Machine Learning Research, 306:53926–53943. https://proceedings.mlr.press/v306/jin26e.html

[18] Lyu, Y., Wang, C., Huang, J., and Xu, T. (2026). Student-Centered Distillation Narrows the Agentic Gap Between Small and Large LLMs. ICML 2026, Proceedings of Machine Learning Research, 306:84006–84029. https://proceedings.mlr.press/v306/lyu26j.html

[19] Li, G., Zheng, M., Song, M., Liu, R., Yang, T., Sun, J., Zhong, Q., Zhang, D., Guo, H., Fang, J., and Wang, J. (2026). On-Policy Distillation with Curriculum Turn-level Guidance for Multi-turn Agents. EMNLP 2026, 32539–32555. https://aclanthology.org/2026.emnlp-main.1424/

[20] Kim, Y., Shin, D., Kang, M., Na, B., and Moon, I.-C. (2026). Distillation of Large Language Models via Concrete Score Matching. ICLR 2026. https://proceedings.iclr.cc/paper_files/paper/2026/hash/1bfc9f74afa91b9b8add5a97a97001a1-Abstract-Conference.html

[21] Górski, F., and Czyżewski, A. (2026). ADMEDTAGGER: an annotation framework for distillation of expert knowledge for the Polish medical language. Scientific Reports, 16, 23587. https://www.nature.com/articles/s41598-026-54167-y

[22] Shen, S., and Zhang, X. (2026). Preference-Aligned Data Synthesis for Fine-Grained Intent Detection. EMNLP 2026, 3742–3758. https://aclanthology.org/2026.emnlp-main.163/

[23] Castillo-López, G., Lombard, A., de Chalendar, G., and Semmar, N. (2026). Leveraging Turn-taking Dynamics for Intent Recognition in Multi-party Conversations. EMNLP 2026 Industry Track, 2717–2731. https://aclanthology.org/2026.emnlp-industry.160/

## Appendix A. Teacher labeling specification

**Input JSON:** {"utterance":"...", "history":[{"role":"...", "text":"..."}], "workflow_state":"...", "allowed_labels":["GUIDANCE","PRECONSULT","SERVICE","OTHER"]}. Explicitly exclude identity, diagnosis, identifiers, and unnecessary patient data. Teacher prompt must define the four routing categories and require evidence based on literal text/context.

**Output JSON:** {"primary_intent":"...", "secondary_intents":[], "needs_clarification":true/false, "ambiguity_reason":"...", "risk_trigger":true/false, "evidence_span":"quoted input text", "policy_action":"ROUTE|CLARIFY|ESCALATE"}. If teacher reports numeric confidence, preserve provenance and mark it uncalibrated unless independently calibrated. Do not include unrestricted diagnostic advice.

**Verification checks:** Schema valid; allowed label; evidence span appears in source; no data leakage; no hidden real patient identifiers; no unsupported certainty; risk cases audited; multiple-teacher disagreement queued for review. Keep raw teacher response and parser status only under access-controlled logs.

## Appendix B. Reproducibility checklist

1. Lock taxonomy, annotation manual, and router API schema before annotation. 2. Reserve disjoint test sets by patient/session, site, and time before any teacher labeling or synthetic generation. 3. Record teacher service names, immutable model versions where supported, prompts, decoding parameters, availability, rate limits and cost. 4. Build encoder and naive SFT baselines first. 5. Run teacher-quality checks on expert gold before pseudo-label scale-up. 6. Obtain IRB/institutional approval and private-model terms before touching real data. 7. Store pseudo-labels with source and review metadata; prevent pseudo-labels from overriding expert gold. 8. Train at least five seeds for key comparisons; log invalid generation and truncation failures. 9. Lock calibration and escalation thresholds on development only. 10. Report per-class/per-site/per-seed metrics, CIs, latency, and negative results. 11. Register code/checkpoints/annotation releases only if data rights permit. 12. Never claim OPD unless student-generated trajectories and token-level teacher scoring genuinely occur.
