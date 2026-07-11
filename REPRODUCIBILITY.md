# Reproduction guide

## Level 1: verify reported values

```bash
python scripts/verify_package.py
```

This is deterministic, CPU-only, and reads the packaged outputs.

## Level 2: rerun saved-output analyses

```bash
pip install -r requirements-analysis.txt
python scripts/reproduce_saved_results.py --output-dir reproduced
```

This copies source artifacts into `reproduced/` and reruns reference-display,
mechanistic-control, and steering summaries without altering packaged data.
Runtime depends on CPU core count and available RAM; no model download is
required because `self_answer_activations.npz` is included.

## Level 3: regenerate model outputs

The end-to-end pipeline is GPU-intensive and was run on an NVIDIA A100 40 GB.
Install the GPU environment:

```bash
pip install -r scripts/requirements-gpu.txt
hf auth login
```

Then run from the repository root:

```bash
python scripts/level25_full_experiment_pipeline.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output-root reproduced_full \
  --candidates-csv data/inputs/candidates_qwen_corpus.csv
```

The primary evaluator revision recorded by the run is
`a09a35458c702b33eeacc393d103063234e8bc28`. The pipeline records runtime,
configuration, git state, and stage-completion markers. Exact numerical
identity is not guaranteed across CUDA, vLLM, Transformers, or GPU versions,
even with fixed seeds.

The historical behavioral-validation and cross-model runs did not record
immutable revisions for every model. Their saved outputs are included for
auditability, but they support result verification rather than bitwise reruns.

## Paper build

From `paper/`:

```bash
pdflatex paperMechConcise.tex
bibtex paperMechConcise
pdflatex paperMechConcise.tex
pdflatex paperMechConcise.tex
```

The paper source uses standard LaTeX packages and the included figure PDFs.

