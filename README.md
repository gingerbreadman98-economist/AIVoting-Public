# Cardinal-Looking Ballots, Comparative Representations

Reproducibility package for `paper/paperMechConcise.tex`.

The paper studies Absolute Allocation ballots as a behavioral assay for LLM
evaluators. Its central result is a measurement-representation dissociation:
Qwen's expressed ballot contains signed, cardinal-looking behavior, while the
tested linear readout is principally comparative and bipolar. Semantic probe
directions causally influence expressed allocations, subject to the ballot
normalization limitations documented in the paper.

## Quick verification

The fastest check uses only the Python standard library and the saved CSVs:

```bash
python scripts/verify_package.py
```

This verifies required artifacts and recomputes the headline values used in
the paper. It does not download models or require a GPU.

To rerun the saved-output analyses:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements-analysis.txt
python scripts/reproduce_saved_results.py --output-dir reproduced
```

## Repository map

- `paper/`: concise LaTeX source, bibliography, and referenced figures.
- `data/inputs/`: frozen prompt/candidate inputs.
- `data/primary_votes/`: Qwen hidden anchor, visible reference, and hidden
  placebo runs.
- `data/reference_display/`: paired reference-display summaries.
- `data/primary_mechanistic/`: activation matrix, row labels, probe outputs,
  residual tests, dimensionality controls, and length controls.
- `data/steering/`: fresh-baseline Qwen steering outputs and derived analyses.
- `data/cross_model/`: Llama and Gemma behavioral elicitation runs.
- `data/behavioral_validation/`: 14B-validated aggregation experiment.
- `scripts/`: generation, extraction, analysis, and verification code.
- `metadata/`: software/model provenance and SHA-256 manifest.

See `CLAIM_MAP.md` for the source of every headline result,
`DATA_DICTIONARY.md` for terminology, and `REPRODUCIBILITY.md` for GPU
regeneration instructions.

## Scope and limitations

The package includes saved activations so the CPU analyses can be rerun without
model inference. It excludes model weights, Hugging Face caches, virtual
environments, credentials, and duplicate JSONL copies of CSV data. The Qwen
evaluator revision is pinned where recorded. Historical Llama, Gemma, candidate
generator, and external-judge runs did not record immutable revisions; this is
disclosed in `metadata/model_revisions.csv` and limits bitwise regeneration.

No Hugging Face token is included. Gated-model regeneration requires users to
authenticate independently.

