# Technical handoff for Nayang's Claude / Codex

Handoff date: 2026-11-24. Repository owner: terrense. Nayang is the incoming project lead.

## Goal and scope

Build a small, reliable Chinese hospital intent router. The four labels select workflows, not diagnoses. Build a traceable distillation dataset, then evaluate whether Qwen3-8B SFT improves on short prompting and compact classifiers. Do not assume 8B or a sophisticated KD objective must win.

The current work is data/experiment preparation. The owner previously reserved actual training/deployment for later; do not start either just because training scripts are present.

## Directory map

| Path | What it contains |
|---|---|
| root Python files | First working annotation/prediction/splitting/export pipeline and optional future QLoRA runner |
| config/ | Root hospital-intent-v0.2-draft policy and candidate-generation prompt |
| data/handoff_v0.2/ | 156 candidates, teacher references, old blind review packets, manifest |
| data/pilot_* and reports/ | Frozen v0.1 development runs, raw probabilities, provisional scores, format fixtures |
| 实验准备/ | Newer research schema, action-aware annotation, calibration, statistics, acquisition and CPU baseline entry point |
| 实验准备/data/annotation_*.v1.jsonl | Preferred new human-review packets, opaque ids, no teacher answers |
| 实验准备/sources/ | User-provided research material and visible shared-chat snapshot |
| business_context/hgdoctor_intent_snapshot/ | 13-file read-only snapshot of the original intent module/API/config |
| archive/originals/ | Exact user-supplied ZIP and review |
| archive/prior_deliveries/ | Historical ZIP deliveries, not the current working instructions |
| audit/ | Upload scope, hashes and restored binary provenance |

## Three policy versions — do not silently mix them

- `hospital-intent-v0.1-draft`: frozen pilot. It incorrectly treated isolated fragments asother; retain only as history.
- `hospital-intent-v0.2-draft`: root pipeline; isolated fragments are kept in a clarification pool instead ofother.
- `care-prep-v1`: research records. Labels remain triage/preconsult/info/other, with separate secondary intents, action, evidence and review state. Imported teacher labels are references; human labels remain null until review.

`GUIDANCE/PRECONSULT/SERVICE/OTHER` in the manuscript map to `triage/preconsult/info/other`. Do not interpret physical directions to a named clinic as triage.

## Known state

156 current candidates; zero approved human gold. 153 associated Jev vectors pass structural validation; none is independently calibrated/approved for soft KD. 97 reported vectors are exact one-hot; add hard-label and entropy/smoothing controls before claiming dark-knowledge gains.

The 2026-11-18 Jev endpoint returned `jev-1.14.0`. An immutable upstream revision was not verified. The last recorded balance11844 is historical. Do not claim current key health or credit balance without a new explicit bounded call.

The existing local model used for pilot comparisons was Qwen3-VL-8B-Instruct. The desired student is text Qwen3-8B. The generator self-label, local classifier vote and ChatGPT-authored proposal are distinct provenance signals, not three independent gold annotators.

The 32-example comparison was assistant-authored development data. Original system matched25/32, short classifier32/32, Jev32/32; all are provisional and must not become headline evaluation. Old/new paths also differ in supplied workflow state and output responsibility.

## Where to resume

1. Run both offline suites and data validation; preserve outputs under a new run id rather than overwriting historical reports.
2. Help Nayang and a second human label156 examples, add44 independent cases, adjudicate200 and freeze scope. Agent suggestions must not be represented as human labels.
3. Resolve personal-medication/general-medical-QA scope, external site, real time cutoff and professional escalation review. Keep unavailable test partitions empty and explicitly blocked.
4. Create a training-only candidate pool and source/semantic-family ledger. Existing60 generated records are quarantined, not accepted volume.
5. Measure teacher quality on training-side human gold, then scale bounded labeling. Separate candidate counts, accepted pseudo labels, human labels, judgment counts and money.
6. Once Nayang requests model experiments, use the run matrix and fixed data/teacher budgets. Supply complete class probabilities and checkpoint fingerprints for calibration; raw generated labels alone do not support soft KL or risk-coverage analysis.

## Credential handoff

The live key must not be in Git history. Repository Actions Secret name: `JEVSTATION_API_KEY`. `docs/CREDENTIALS.md` explains the manual workflow and local setup. Secrets are not readable back from GitHub; do not invent a retrieval mechanism or print them into an artifact. No automatic workflow spends credits.

The old generator can use an existing hgdoctor Docker service, but Nayang does not need that service for offline checks. For new generation use her own configured compatible model. No GPT key is included.

## Model and material caveats

- Original hgdoctor code bundled classification, symptom extraction, question splitting and reply generation in ~11,968-character prompts. A short classifier is a baseline; deleting old downstream extraction without adapters would break the product.
- The source `.docx` on the original workstation was protected/opaque. A valid58343-byte Word file was recovered directly from the owner's original ZIP. Keep the protected copy only for provenance; do not replace the restored Git object with a `%TSD` file.
- Original manifests describe historical snapshots. `audit/UPLOAD_MANIFEST.json` describes this handoff. A changed restored Word binary can legitimately differ from a historical snapshot hash.
- No repository-wide license was inferred from public visibility. Keep source attribution and do not add a blanket license over third-party research material without an owner decision.
