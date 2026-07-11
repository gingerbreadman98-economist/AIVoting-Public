#!/usr/bin/env python3
"""
One-shot, set-and-forget pipeline for the full Absolute Allocation experiment
on a single evaluator model. Designed for a rented GPU box under tmux: launch
it, walk away, come back to finished, analysis-ready results.

Given one voting model it runs, in order:

  Stage 1  vote_hidden_anchor    Hidden-reference vote run (anchor A, seed s0).
                                 Generates candidates unless --candidates-csv.
  Stage 2  vote_visible          Visible-reference vote run (treatment V,
                                 seed s1) with EXACT paired inputs from A
                                 (same self-answers, candidate order, ballot
                                 field order). Only visibility + seed differ.
  Stage 3  vote_hidden_placebo   Hidden re-run (placebo B, seed s1) with the
                                 same paired inputs. Differs from A only by
                                 sampling seed -> measures the noise floor.
  Stage 4  anchoring_analysis    Winner and voter-level changes: effect (V vs
                                 A), placebo floor (B vs A), placebo-corrected
                                 effect, plus a separately labeled normalized-
                                 ballot recovery sensitivity analysis.
  Stage 5  mech_extract_hidden   Activation extraction + probe sweep on A.
  Stage 6  mech_extract_visible  Activation extraction + probe sweep on V.
  Stage 7  offline_<cond>        Offline analysis battery on each mech dir:
                                   - extra_offline (probes, position baselines,
                                     residual-beyond-rank, pairwise, no-repair)
                                   - direction/position controls (split-half
                                     reliability, position-matched AUC)
                                   - length controls (length-only baseline,
                                     length-matched AUC, residual+length)
                                   - dimensionality tests (1-D sufficiency,
                                     neutrality decodability)
                                   - cosine reliability sensitivity (C grid)
  Stage 8  steering_<cond>       Causal activation steering (full signed +
                                 random grid, fresh-baseline targets, plus
                                 zero-intervention baseline replicates as a
                                 per-ballot generation-noise floor) on the
                                 condition chosen by --steering-condition,
                                 followed by detailed movement, spillover,
                                 clustered regression, and intensity analyses.
  Stage 9  summary               SUMMARY.md digest of every headline number.

Resume: every stage writes a marker in <output-root>/pipeline_state/. Re-running
the same command skips completed stages, so a crashed or preempted instance can
be resumed by simply re-launching. Use --force-stage <name> (repeatable) or
delete markers to redo a stage. Logs stream to <output-root>/logs/<stage>.log.

Example (Llama, layer-18 analysis, steering on the hidden condition):

  python level25_full_experiment_pipeline.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output-root llama31_full_experiment \
    --layers 14,18,22 --analysis-layer 18 \
    --steering-condition hidden \
    --gpu-memory-utilization 0.90

Rough wall-clock on a single A100/4090-class GPU, 100 prompts x 4 evaluators:
vote runs ~20-40 min each; extraction+probe sweep ~1-3 h per condition;
offline battery ~1-2 h per condition (CPU); steering ~4-8 h (14 interventions
x 3 strengths x 80 ballots + baselines). Budget roughly a day end to end.

The default ``--layers fast`` policy extracts one depth-matched layer: layer 18
for a 32-block model and layer 16 for a 28-block model. Broader comma-separated
layer sweeps remain available explicitly.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE_PROTOCOL_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full anchoring + mechanistic + steering experiment for one model."
    )
    parser.add_argument("--model", required=True, help="Evaluator model (HF id).")
    parser.add_argument(
        "--model-revision",
        default="main",
        help="HF revision to resolve and pin. The resolved commit is recorded and used.",
    )
    parser.add_argument("--output-root", default="", help="Root dir for all artifacts.")
    parser.add_argument("--candidate-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--candidate-model-revision", default="main")
    parser.add_argument("--prompts-csv", default="")
    parser.add_argument(
        "--candidates-csv",
        default="",
        help="Reuse an existing candidates.csv (recommended for cross-model comparability).",
    )
    parser.add_argument("--max-prompts", type=int, default=0, help="0 = all prompts.")
    parser.add_argument("--evaluator-repeats", type=int, default=4)
    parser.add_argument("--evaluator-temperature", type=float, default=0.05)
    parser.add_argument("--evaluator-top-p", type=float, default=0.95)
    parser.add_argument("--anchor-seed", type=int, default=7)
    parser.add_argument("--treatment-seed", type=int, default=11,
                        help="Seed for BOTH the visible run and the hidden placebo, so each "
                             "differs from the anchor by the same seed gap.")
    # Ballot construction.
    parser.add_argument("--strict-borda", action="store_true",
                        help="Force strict rankings (no ties). Default off to preserve the "
                             "tie-refinement analyses.")
    parser.add_argument("--enforce-exact-allocation", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--allocation-max-retries", type=int, default=3)
    parser.add_argument("--randomize-ballot-field-order", action=argparse.BooleanOptionalAction,
                        default=True)
    # Mechanistic settings.
    parser.add_argument(
        "--layers",
        default="fast",
        help="'fast' selects layer 18/32 or 16/28; otherwise use a comma list.",
    )
    parser.add_argument(
        "--analysis-layer",
        type=int,
        default=0,
        help="Layer for offline tests/steering. Zero uses the fast depth match.",
    )
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--cosine-repeats", type=int, default=50)
    # Steering settings.
    parser.add_argument("--steering-condition", choices=["hidden", "visible", "both", "skip"],
                        default="hidden")
    parser.add_argument("--steering-max-ballots", type=int, default=80)
    parser.add_argument("--steering-max-prompts", type=int, default=25)
    parser.add_argument(
        "--steering-baseline-replicates",
        type=int,
        default=3,
        help=(
            "Total unsteered baseline generations per steering ballot. Replicate 0 "
            "drives target selection and deltas; the rest estimate per-ballot "
            "generation noise at the same temperature (~+5%% generations at 3)."
        ),
    )
    parser.add_argument("--steering-strengths", default="0.05,0.10,0.20")
    parser.add_argument("--steering-seed", type=int, default=13)
    parser.add_argument(
        "--steering-interventions",
        default=(
            "help_to_bottom,help_to_top,hurt_to_bottom,hurt_to_top,"
            "neg_help_to_bottom,neg_help_to_top,neg_hurt_to_bottom,neg_hurt_to_top,"
            "random_to_bottom,random_to_top,random2_to_bottom,random2_to_top,"
            "random3_to_bottom,random3_to_top"
        ),
    )
    # GPU / runtime pass-through.
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", default="eager",
                        choices=["", "eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help=(
            "Fast one-shot: run only the GPU-bound stages (vote runs, activation "
            "extraction, steering) plus the cheap anchoring analysis, skip the "
            "CPU offline probe battery, then zip the mech dirs and emit "
            "run_offline_locally.sh so the experimenter finishes the analysis "
            "off-box."
        ),
    )
    parser.add_argument("--allow-provenance-change", action="store_true")
    parser.add_argument("--force-stage", action="append", default=[],
                        help="Redo this stage even if its marker exists (repeatable).")
    args = parser.parse_args()
    return args


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        model_slug = args.model.replace("/", "_").replace(":", "_")
        self.root = Path(args.output_root) if args.output_root else Path(
            f"level25_pipeline_{model_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.state_dir = self.root / "pipeline_state"
        self.log_dir = self.root / "logs"
        for d in (self.root, self.state_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.resolved_revision = ""
        self.resolved_candidate_revision = ""
        self.model_depth = 0
        self.resolve_model_and_layers()
        # Canonical artifact locations.
        self.run_a = self.root / "vote_hidden_anchor"
        self.run_v = self.root / "vote_visible"
        self.run_b = self.root / "vote_hidden_placebo"
        self.anchoring_dir = self.root / "anchoring_analysis"
        self.anchoring_detail_dir = self.root / "anchoring_voter_level"
        self.anchoring_floor_dir = self.root / "anchoring_voter_level_placebo"
        self.recovery_dir = self.root / "allocation_recovery_sensitivity"
        self.mech = {"hidden": self.root / "mech_hidden", "visible": self.root / "mech_visible"}
        self.steer_dir = {c: self.root / f"steering_{c}" for c in ("hidden", "visible")}
        operational = {"force_stage", "preflight_only", "dry_run", "allow_provenance_change"}
        pipeline_config = {
            **{key: value for key, value in vars(args).items() if key not in operational},
            "resolved_model_revision": self.resolved_revision,
            "resolved_candidate_model_revision": self.resolved_candidate_revision,
            "model_depth": self.model_depth,
            "resolved_layers": args.layers,
            "resolved_analysis_layer": args.analysis_layer,
        }
        config_path = self.root / "pipeline_config.json"
        if config_path.exists():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if existing != pipeline_config:
                raise ValueError(
                    "Scientific pipeline configuration differs from the existing output root. "
                    "Use a new --output-root instead of mixing protocols."
                )
        else:
            config_path.write_text(
                json.dumps(pipeline_config, indent=2, default=str), encoding="utf-8"
            )

    def resolve_model_and_layers(self) -> None:
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                self.args.model,
                revision=self.args.model_revision,
                trust_remote_code=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Could not resolve model config/revision: {exc}") from exc
        depth = int(
            getattr(config, "num_hidden_layers", 0)
            or getattr(getattr(config, "text_config", None), "num_hidden_layers", 0)
            or 0
        )
        if depth <= 0:
            raise ValueError("Could not determine transformer-block count from model config.")
        fast_layer = {32: 18, 28: 16}.get(depth)
        if str(self.args.layers).strip().lower() == "fast":
            if fast_layer is None:
                raise ValueError(
                    f"Fast layer policy is defined for 28/32-block models, but {self.args.model} "
                    f"reports {depth}. Pass --layers and --analysis-layer explicitly."
                )
            self.args.layers = str(fast_layer)
        layer_list = [int(x) for x in str(self.args.layers).split(",") if x.strip()]
        if not self.args.analysis_layer:
            if fast_layer is None:
                raise ValueError("Pass --analysis-layer explicitly for this model depth.")
            self.args.analysis_layer = fast_layer
        if self.args.analysis_layer not in layer_list:
            raise ValueError(
                f"--analysis-layer {self.args.analysis_layer} must be in --layers {layer_list}."
            )
        if any(layer < 0 or layer >= depth for layer in layer_list):
            raise ValueError(f"Requested layers {layer_list} are invalid for {depth} blocks.")
        self.model_depth = depth
        self.resolved_revision = str(getattr(config, "_commit_hash", "") or self.args.model_revision)
        if not self.args.candidates_csv:
            candidate_config = AutoConfig.from_pretrained(
                self.args.candidate_model,
                revision=self.args.candidate_model_revision,
                trust_remote_code=True,
            )
            self.resolved_candidate_revision = str(
                getattr(candidate_config, "_commit_hash", "") or self.args.candidate_model_revision
            )
            self.args.candidate_model_revision = self.resolved_candidate_revision

    # ---------- stage machinery ----------
    def marker(self, stage: str) -> Path:
        return self.state_dir / f"{stage}.done"

    def run_stage(self, stage: str, cmd: list[str], required_outputs: list[Path]) -> None:
        if self.marker(stage).exists() and stage not in self.args.force_stage:
            if all(p.exists() for p in required_outputs):
                print(f"[pipeline] SKIP {stage} (marker + outputs present)", flush=True)
                return
            print(f"[pipeline] marker for {stage} exists but outputs missing; re-running", flush=True)
        print(f"[pipeline] RUN  {stage}\n           {' '.join(cmd)}", flush=True)
        if self.args.dry_run:
            return
        log_path = self.log_dir / f"{stage}.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n===== {datetime.now().isoformat()} :: {' '.join(cmd)}\n")
            log.flush()
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE))
        if result.returncode != 0:
            print(f"[pipeline] FAILED {stage} (exit {result.returncode}); see {log_path}", flush=True)
            sys.exit(result.returncode)
        missing = [str(p) for p in required_outputs if not p.exists() or (p.is_file() and p.stat().st_size == 0)]
        if missing:
            print(f"[pipeline] FAILED {stage}: expected outputs missing: {missing}", flush=True)
            sys.exit(1)
        self.marker(stage).write_text(datetime.now().isoformat(), encoding="utf-8")
        print(f"[pipeline] DONE {stage}", flush=True)

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def capture(command: list[str]) -> str:
        try:
            return subprocess.run(
                command,
                cwd=str(HERE),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            ).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            return f"unavailable: {exc}"

    @staticmethod
    def script_protocol_version(path: Path) -> int | None:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(isinstance(target, ast.Name) and target.id == "PIPELINE_PROTOCOL_VERSION" for target in node.targets):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                    return int(node.value.value)
        return None

    def preflight(self) -> None:
        required_scripts = [
            "level25_self_answer_vote_vLLM.py",
            "level25_self_answer_activation_analysis.py",
            "level25_anchoring_placebo_analysis.py",
            "analyze_reference_anchoring.py",
            "level25_recover_allocation_ballots.py",
            "level25_causal_activation_steering.py",
            "analyze_causal_steering_outputs.py",
            "analyze_causal_steering_spillovers.py",
            "analyze_causal_steering_regressions.py",
            "level25_steering_intensity_analysis.py",
            # Offline battery + shared prompt/parsing modules; a missing one of
            # these would otherwise only fail after all GPU stages completed.
            "level25_extra_offline_analysis.py",
            "level25_direction_position_controls.py",
            "level25_length_controls.py",
            "level25_dimensionality_tests.py",
            "level25_cosine_reliability_sensitivity.py",
            "level25_ballot_prompt.py",
            "level1_direct_vote_eval.py",
            "level1_direct_vote_eval_vLLM.py",
        ]
        missing = [name for name in required_scripts if not (HERE / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing pipeline scripts: {missing}")
        protocol_scripts = [
            "level25_self_answer_vote_vLLM.py",
            "level25_self_answer_activation_analysis.py",
            "level25_causal_activation_steering.py",
            "level25_recover_allocation_ballots.py",
        ]
        incompatible = {
            name: self.script_protocol_version(HERE / name)
            for name in protocol_scripts
            if self.script_protocol_version(HERE / name) != PIPELINE_PROTOCOL_VERSION
        }
        if incompatible:
            raise RuntimeError(
                "Mixed experiment script versions detected before launch. Expected protocol "
                f"{PIPELINE_PROTOCOL_VERSION}, found {incompatible}. Upload the complete script set."
            )
        inputs = {}
        for label, raw in (("prompts_csv", self.args.prompts_csv), ("candidates_csv", self.args.candidates_csv)):
            if raw:
                path = Path(raw).resolve()
                if not path.exists() or path.stat().st_size == 0:
                    raise FileNotFoundError(path)
                inputs[label] = {"path": str(path), "bytes": path.stat().st_size, "sha256": self.sha256(path)}
        if not self.args.candidates_csv:
            print(
                "[pipeline] WARNING: no --candidates-csv supplied; candidates will be generated "
                "once in the anchor and then frozen for paired runs.",
                flush=True,
            )
        free = shutil.disk_usage(self.root.resolve()).free
        if free < 20 * 1024**3:
            raise RuntimeError(f"Less than 20 GiB free at output root ({free / 1024**3:.1f} GiB).")
        runtime_check = "skipped (dry run)"
        if not self.args.dry_run:
            probe = (
                "import json, torch, vllm, transformers, sklearn, pandas;"
                "assert torch.cuda.is_available(), 'CUDA is not available';"
                "print(json.dumps({"
                "'torch': torch.__version__, 'vllm': vllm.__version__,"
                "'transformers': transformers.__version__,"
                "'sklearn': sklearn.__version__, 'pandas': pandas.__version__,"
                "'gpu': torch.cuda.get_device_name(0),"
                "'gpu_count': torch.cuda.device_count()}))"
            )
            result = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=str(HERE), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Runtime preflight failed (torch/vllm import or CUDA):\n"
                    + result.stdout.strip()
                )
            runtime_check = json.loads(result.stdout.strip().splitlines()[-1])
            print(f"[pipeline] runtime check: {runtime_check}", flush=True)
        provenance = {
            "runtime_check": runtime_check,
            "created_at": datetime.now().isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "cwd": str(HERE),
            "model": self.args.model,
            "requested_revision": self.args.model_revision,
            "resolved_revision": self.resolved_revision,
            "candidate_model": self.args.candidate_model,
            "resolved_candidate_revision": self.resolved_candidate_revision,
            "model_depth": self.model_depth,
            "layers": self.args.layers,
            "analysis_layer": self.args.analysis_layer,
            "inputs": inputs,
            "git_head": self.capture(["git", "rev-parse", "HEAD"]),
            "git_status": self.capture(["git", "status", "--short"]),
            "pip_freeze": self.capture([sys.executable, "-m", "pip", "freeze"]).splitlines(),
            "nvidia_smi": self.capture(["nvidia-smi"]),
            "environment": {
                key: os.environ.get(key, "")
                for key in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "CUDA_VISIBLE_DEVICES")
            },
            "source_sha256": {
                path.name: self.sha256(path)
                for path in sorted(HERE.glob("*.py"))
            },
        }
        (self.root / "provenance_latest.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )
        initial_provenance = self.root / "provenance_initial.json"
        if initial_provenance.exists():
            initial = json.loads(initial_provenance.read_text(encoding="utf-8"))
            changed = [
                key for key in ("resolved_revision", "inputs", "source_sha256")
                if initial.get(key) != provenance.get(key)
            ]
            if changed and not self.args.allow_provenance_change:
                raise ValueError(
                    f"Provenance changed on resume ({', '.join(changed)}). Use a new output "
                    "root, or explicitly pass --allow-provenance-change after auditing it."
                )
        else:
            initial_provenance.write_text(
                json.dumps(provenance, indent=2), encoding="utf-8"
            )
        analysis_plan = {
            "frozen_before_generation": True,
            "primary_ballot_population": "strict-valid ballots common to paired conditions",
            "sensitivity_population": "strict-valid plus labeled L1-normalized recoverable ballots",
            "anchoring_design": {
                "anchor": f"hidden reference, seed {self.args.anchor_seed}",
                "treatment": f"visible reference, seed {self.args.treatment_seed}",
                "placebo": f"hidden reference, seed {self.args.treatment_seed}",
                "primary_unit": "prompt for election outcomes; ballot/candidate for voter outcomes",
                "primary_contrast": "visible seed s1 versus hidden-placebo seed s1",
                "noise_diagnostic": "anchor hidden s0 versus hidden-placebo s1",
                "primary_outcomes": [
                    "allocation winner change",
                    "best-pick winner change",
                    "Borda winner change",
                    "candidate allocation change",
                    "allocation polarity change",
                ],
            },
            "mechanistic_design": {
                "layer_policy": "architecture-depth match selected before outcome inspection",
                "layers": self.args.layers,
                "analysis_layer": self.args.analysis_layer,
                "cross_validation_group": "prompt_id",
                "context_mode": "exact_ballot",
            },
            "steering_design": {
                "condition": self.args.steering_condition,
                "strengths": self.args.steering_strengths,
                "interventions": self.args.steering_interventions,
                "target_source": "fresh_baseline",
                "random_controls": ["random", "random2", "random3"],
                "baseline_replicates": self.args.steering_baseline_replicates,
                "cluster_unit": "ballot_id",
            },
            "exploratory_label": "Probe hyperparameter sweeps and any post hoc subgroup analyses are exploratory.",
        }
        analysis_plan_path = self.root / "analysis_plan.json"
        if analysis_plan_path.exists():
            existing_plan = json.loads(analysis_plan_path.read_text(encoding="utf-8"))
            if existing_plan != analysis_plan:
                raise ValueError("Existing analysis_plan.json differs; use a new output root.")
        else:
            analysis_plan_path.write_text(
                json.dumps(analysis_plan, indent=2), encoding="utf-8"
            )
        if provenance["git_status"]:
            print("[pipeline] WARNING: working tree is dirty; provenance records the diff state.", flush=True)

    def validate_vote_runs(self) -> None:
        import pandas as pd

        runs = {"anchor": self.run_a, "visible": self.run_v, "placebo": self.run_b}
        design_cols = ["prompt_id", "evaluator_id", "candidate_display_order", "ballot_field_order"]
        anchor_diagnostics_preview = pd.read_csv(self.run_a / "allocation_validation_diagnostics.csv")
        legacy_complete_case = any(
            column not in anchor_diagnostics_preview.columns for column in design_cols
        )
        frames = {}
        rows = []
        anchor_parsed_ballots = 0
        for name, run in runs.items():
            prompts = pd.read_csv(run / "prompts.csv")
            candidates = pd.read_csv(run / "candidates.csv")
            votes = pd.read_csv(run / "direct_votes.csv")
            diagnostics = pd.read_csv(run / "allocation_validation_diagnostics.csv")
            self_answers = pd.read_csv(run / "self_answers.csv")
            expected_ballots = len(prompts) * self.args.evaluator_repeats
            ballot_keys = votes[["prompt_id", "evaluator_id"]].drop_duplicates()
            expected_attempts = (
                anchor_parsed_ballots
                if legacy_complete_case and name != "anchor"
                else expected_ballots
            )
            if len(diagnostics) != expected_attempts:
                raise ValueError(
                    f"{name}: expected {expected_attempts} ballot diagnostics, got {len(diagnostics)}"
                )
            if len(self_answers) != expected_ballots:
                raise ValueError(f"{name}: expected {expected_ballots} self-answers, got {len(self_answers)}")
            counts = votes.groupby(["prompt_id", "evaluator_id"]).size()
            if not counts.eq(4).all():
                raise ValueError(f"{name}: parsed ballots do not contain exactly four candidate rows")
            if votes.duplicated(["prompt_id", "evaluator_id", "candidate_id"]).any():
                raise ValueError(f"{name}: duplicate candidate rows detected")
            finite = pd.to_numeric(votes["allocation"], errors="coerce").notna().all()
            if not finite:
                raise ValueError(f"{name}: nonfinite parsed allocations detected")
            if all(column in diagnostics.columns for column in design_cols):
                design = diagnostics[design_cols].copy()
                design_source = "all_attempted_ballots"
            else:
                missing_vote_design = [column for column in design_cols if column not in votes.columns]
                if missing_vote_design:
                    raise ValueError(f"{name}: no archived ballot design; missing {missing_vote_design}")
                design = votes.drop_duplicates(["prompt_id", "evaluator_id"])[design_cols].copy()
                design_source = "parsed_ballots_only_legacy"
            design = design.sort_values(["prompt_id", "evaluator_id"]).reset_index(drop=True)
            frames[name] = (candidates, self_answers, votes, design)
            if name == "anchor":
                anchor_parsed_ballots = len(ballot_keys)
            rows.append(
                {
                    "run": name,
                    "prompts": len(prompts),
                    "candidates": len(candidates),
                    "expected_ballots": expected_ballots,
                    "parsed_ballots": len(ballot_keys),
                    "invalid_after_retries": expected_ballots - len(ballot_keys),
                    "strict_parse_rate": len(ballot_keys) / expected_ballots,
                    "attempted_ballots": len(diagnostics),
                    "ballot_design_source": design_source,
                    "legacy_complete_case_analysis": legacy_complete_case,
                    "anchor_contexts_omitted_from_paired_runs": (
                        expected_ballots - anchor_parsed_ballots
                        if legacy_complete_case and name != "anchor"
                        else 0
                    ),
                }
            )
        base_candidates = frames["anchor"][0].sort_values(list(frames["anchor"][0].columns)).reset_index(drop=True)
        base_self = frames["anchor"][1].sort_values(["prompt_id", "evaluator_id"]).reset_index(drop=True)
        for name in ("visible", "placebo"):
            other_candidates = frames[name][0].sort_values(list(frames[name][0].columns)).reset_index(drop=True)
            if not base_candidates.equals(other_candidates):
                raise ValueError(f"{name}: candidates differ from anchor")
            other_self = frames[name][1].sort_values(["prompt_id", "evaluator_id"]).reset_index(drop=True)
            common_cols = [c for c in ("prompt_id", "evaluator_id", "self_answer") if c in base_self.columns]
            if not base_self[common_cols].equals(other_self[common_cols]):
                raise ValueError(f"{name}: self-answers differ from anchor")
            design_pair = frames["anchor"][3].merge(
                frames[name][3],
                on=["prompt_id", "evaluator_id"],
                suffixes=("_anchor", f"_{name}"),
                validate="one_to_one",
            )
            if len(design_pair) != len(frames[name][3]):
                raise ValueError(f"{name}: contains ballot designs absent from anchor")
            for column in ("candidate_display_order", "ballot_field_order"):
                if not design_pair[f"{column}_anchor"].equals(design_pair[f"{column}_{name}"]):
                    raise ValueError(f"{name}: {column} differs from anchor")
        qc = pd.DataFrame(rows)
        qc.to_csv(self.root / "vote_run_quality_control.csv", index=False)
        if legacy_complete_case:
            print(
                "[pipeline] WARNING: continuing as a legacy complete-case analysis. "
                f"{len(anchor_diagnostics_preview) - anchor_parsed_ballots} invalid anchor "
                "contexts lacked archived design mappings and were not run in paired conditions.",
                flush=True,
            )

    def validate_mech(self, condition: str) -> None:
        import numpy as np
        import pandas as pd

        mech = self.mech[condition]
        index = pd.read_csv(mech / "activation_row_index.csv")
        with np.load(mech / "self_answer_activations.npz") as arrays:
            candidate = arrays["candidate_activations"]
            layers = arrays["layers"].astype(int).tolist()
        expected_layers = [int(x) for x in self.args.layers.split(",")]
        if candidate.shape[0] != len(index):
            raise ValueError(f"{condition}: activation matrix/index row mismatch")
        if layers != expected_layers:
            raise ValueError(f"{condition}: extracted layers {layers} != requested {expected_layers}")
        if index.duplicated(["prompt_id", "evaluator_id", "candidate_id"]).any():
            raise ValueError(f"{condition}: duplicate activation keys")

    def validate_steering(self, condition: str) -> None:
        import pandas as pd

        out = self.steer_dir[condition]
        votes = pd.read_csv(out / "causal_steering_vote_rows.csv")
        raw = pd.read_csv(out / "causal_steering_raw_outputs.csv")
        if votes.empty or raw.empty:
            raise ValueError(f"{condition}: empty steering output")
        baselines = votes[votes["condition"] == "baseline"]
        if baselines.empty or baselines.groupby("ballot_id").size().ne(4).any():
            raise ValueError(f"{condition}: incomplete steering baselines")
        pd.DataFrame(
            [{
                "condition": condition,
                "ballots": int(baselines["ballot_id"].nunique()),
                "steered_raw_outputs": int((raw["condition"] == "steered").sum()),
                "parse_error_rate": float(raw["parse_error"].fillna("").astype(str).ne("").mean()),
                "recovered_invalid_rate": float(
                    raw.get("allocation_recovered_from_invalid", False).fillna(False).astype(bool).mean()
                ) if "allocation_recovered_from_invalid" in raw else 0.0,
            }]
        ).to_csv(out / "pipeline_quality_control.csv", index=False)

    # ---------- command builders ----------
    def vote_cmd(
        self,
        out_dir: Path,
        hidden: bool,
        seed: int,
        paired: bool,
        reuse_paired_inputs: bool = False,
    ) -> list[str]:
        a = self.args
        cmd = [
            sys.executable, "-u", "level25_self_answer_vote_vLLM.py",
            "--evaluator-models", a.model,
            "--fallback-model", a.model,
            "--model-revision", self.resolved_revision,
            "--candidate-model", a.candidate_model,
            "--candidate-model-revision", a.candidate_model_revision,
            "--normal-evaluator-repeats", str(a.evaluator_repeats),
            "--evaluator-temperature", str(a.evaluator_temperature),
            "--evaluator-top-p", str(a.evaluator_top_p),
            "--judge-repeats", "0",
            "--weak-selector-repeats", "0",
            "--shuffle-evaluator-candidates",
            "--seed", str(seed),
            "--output-dir", str(out_dir),
            "--gpu-memory-utilization", str(a.gpu_memory_utilization),
            "--max-model-len", str(a.max_model_len),
            "--batch-size", str(a.batch_size),
            "--dtype", a.dtype,
            "--tensor-parallel-size", str(a.tensor_parallel_size),
            "--allocation-max-retries", str(a.allocation_max_retries),
        ]
        if a.enforce_eager:
            cmd += ["--enforce-eager"]
        if a.prompts_csv:
            cmd += ["--prompts-csv", a.prompts_csv]
        if a.max_prompts:
            cmd += ["--max-prompts", str(a.max_prompts)]
        if hidden:
            cmd += ["--hide-self-answer-reference"]
        if a.strict_borda:
            cmd += ["--strict-borda"]
        if a.enforce_exact_allocation:
            cmd += ["--enforce-exact-allocation"]
        if a.randomize_ballot_field_order:
            cmd += ["--randomize-ballot-field-order"]
        if paired:
            cmd += [
                "--paired-run-dir", str(self.run_a),
                "--candidates-csv", str(self.run_a / "candidates.csv"),
                "--skip-candidate-generation",
            ]
            if reuse_paired_inputs:
                cmd += ["--reuse-paired-inputs"]
        elif a.candidates_csv:
            cmd += ["--candidates-csv", a.candidates_csv, "--skip-candidate-generation"]
        return cmd

    def extract_cmd(self, run_dir: Path, mech_dir: Path) -> list[str]:
        a = self.args
        cmd = [
            sys.executable, "-u", "level25_self_answer_activation_analysis.py",
            "--level25-output-dir", str(run_dir),
            "--model", a.model,
            "--model-revision", self.resolved_revision,
            "--layers", a.layers,
            "--pooling", a.pooling,
            "--max-model-len", str(a.max_model_len),
            "--dtype", a.dtype,
            "--device", "auto",
            "--attn-implementation", a.attn_implementation,
            "--seed", str(a.anchor_seed),
            "--context-mode", "exact_ballot",
            "--save-activation-matrix",
            "--output-dir", str(mech_dir),
        ]
        if a.max_prompts:
            cmd += ["--max-prompts", str(a.max_prompts)]
        return cmd

    def offline_cmds(self, mech_dir: Path) -> list[tuple[str, list[str], list[Path]]]:
        a = self.args
        L = str(a.analysis_layer)
        tag = f"layer{L}"
        return [
            (
                "extra_offline",
                [sys.executable, "-u", "level25_extra_offline_analysis.py",
                 "--mech-output-dir", str(mech_dir), "--layer", L,
                 "--output-dir", str(mech_dir / f"extra_offline_{tag}")],
                [mech_dir / f"extra_offline_{tag}" / "extra_probe_summary.csv"],
            ),
            (
                "dirpos",
                [sys.executable, "-u", "level25_direction_position_controls.py",
                 "--mech-output-dir", str(mech_dir), "--layer", L,
                 "--output-dir", str(mech_dir / f"dirpos_{tag}")],
                [mech_dir / f"dirpos_{tag}" / "position_matched_control.csv"],
            ),
            (
                "length",
                [sys.executable, "-u", "level25_length_controls.py",
                 "--mech-output-dir", str(mech_dir), "--layer", L,
                 "--output-dir", str(mech_dir / f"length_{tag}")],
                [mech_dir / f"length_{tag}" / "length_matched_control.csv"],
            ),
            (
                "dimensionality",
                [sys.executable, "-u", "level25_dimensionality_tests.py",
                 "--mech-output-dir", str(mech_dir), "--layer", L,
                 "--output-dir", str(mech_dir / f"dim_{tag}")],
                [mech_dir / f"dim_{tag}" / "score_dimensionality.csv"],
            ),
            (
                "cosine_sensitivity",
                [sys.executable, "-u", "level25_cosine_reliability_sensitivity.py",
                 "--mech-output-dir", str(mech_dir), "--layer", L,
                 "--repeats", str(a.cosine_repeats),
                 "--output-dir", str(mech_dir / f"cosine_{tag}")],
                [mech_dir / f"cosine_{tag}" / "cosine_reliability_sensitivity.csv"],
            ),
        ]

    def steering_cmds(self, condition: str) -> list[tuple[str, list[str], list[Path]]]:
        a = self.args
        run_dir = self.run_a if condition == "hidden" else self.run_v
        mech_dir = self.mech[condition]
        out_dir = self.steer_dir[condition]
        steer = [
            sys.executable, "-u", "level25_causal_activation_steering.py",
            "--level25-output-dir", str(run_dir),
            "--mech-output-dir", str(mech_dir),
            "--model", a.model,
            "--model-revision", self.resolved_revision,
            "--layer", str(a.analysis_layer),
            "--max-prompts", str(a.steering_max_prompts),
            "--max-ballots", str(a.steering_max_ballots),
            "--baseline-replicates", str(a.steering_baseline_replicates),
            "--strengths", a.steering_strengths,
            "--interventions", a.steering_interventions,
            "--target-source", "fresh_baseline",
            "--seed", str(a.steering_seed),
            "--max-model-len", str(max(a.max_model_len, 4096)),
            "--dtype", a.dtype,
            "--device", "auto",
            "--attn-implementation", a.attn_implementation,
            "--temperature", str(a.evaluator_temperature),
            "--top-p", str(a.evaluator_top_p),
            "--output-dir", str(out_dir),
        ]
        if a.strict_borda:
            steer += ["--strict-borda"]
        intensity = [
            sys.executable, "-u", "level25_steering_intensity_analysis.py",
            "--steering-output-dir", str(out_dir),
            "--output-dir", str(out_dir / "intensity"),
        ]
        detailed = [sys.executable, "-u", "analyze_causal_steering_outputs.py", str(out_dir)]
        spillovers = [sys.executable, "-u", "analyze_causal_steering_spillovers.py", str(out_dir)]
        regressions = [sys.executable, "-u", "analyze_causal_steering_regressions.py", str(out_dir)]
        steer_outputs = [out_dir / "causal_steering_vote_rows.csv"]
        if a.steering_baseline_replicates > 1:
            steer_outputs.append(out_dir / "causal_steering_baseline_noise.csv")
        return [
            (f"steering_{condition}", steer, steer_outputs),
            (f"steering_detail_{condition}", detailed,
             [out_dir / "causal_steering_detailed_analysis.csv"]),
            (f"steering_spillovers_{condition}", spillovers,
             [out_dir / "causal_steering_per_ballot_middle_spillovers.csv"]),
            (f"steering_regressions_{condition}", regressions,
             [out_dir / "causal_steering_regressions.csv"]),
            (f"steering_intensity_{condition}", intensity,
             [out_dir / "intensity" / "steering_intensity_summary.csv"]),
        ]

    # ---------- summary ----------
    def write_summary(self) -> None:
        import pandas as pd

        lines: list[str] = [
            f"# Absolute Allocation pipeline summary — {self.args.model}",
            f"Generated {datetime.now().isoformat()}",
            f"Analysis layer: {self.args.analysis_layer}; layers extracted: {self.args.layers}",
            "",
        ]

        def section(title: str) -> None:
            lines.append(f"\n## {title}\n")

        def table(path: Path, keep: list[str] | None = None, note: str = "") -> None:
            try:
                df = pd.read_csv(path)
                if keep:
                    df = df[[c for c in keep if c in df.columns]]
                df = df.round(4)
                try:
                    lines.append(df.to_markdown(index=False))
                except ImportError:  # tabulate not installed
                    lines.append("```\n" + df.to_string(index=False) + "\n```")
                if note:
                    lines.append(f"\n_{note}_")
            except Exception as exc:  # noqa: BLE001
                lines.append(f"(missing: {path} — {exc})")

        section("Vote-run quality control (ballot health per run)")
        table(self.root / "vote_run_quality_control.csv")

        section("Anchoring: winner-change rates (effect vs placebo floor)")
        table(
            self.anchoring_dir / "winner_change_rates.csv",
            note=(
                "anchoring_effect = visible vs anchor (treatment + seed noise); "
                "placebo_floor = hidden placebo vs anchor (seed noise only); "
                "visible_vs_placebo_same_seed = single-difference primary contrast."
            ),
        )
        section("Anchoring: placebo-corrected effect (per-prompt McNemar)")
        table(self.anchoring_dir / "corrected_effect_paired.csv")
        section("Anchoring: voter-level candidate changes (strict-valid ballots)")
        table(self.anchoring_dir / "candidate_level_change_summary.csv")
        table(self.anchoring_dir / "ballot_level_change_summary.csv")
        section("Anchoring detail: ballot changes, treatment (A vs V) vs noise floor (A vs B)")
        table(
            self.anchoring_detail_dir / "anchoring_ballot_change_summary.csv",
            note="Treatment contrast: hidden anchor vs visible run.",
        )
        table(
            self.anchoring_floor_dir / "anchoring_ballot_change_summary.csv",
            note=(
                "Noise floor: hidden anchor vs hidden placebo (the 'visible' side of "
                "this table is the placebo run; both conditions are hidden)."
            ),
        )
        section("Allocation recovery sensitivity")
        table(self.recovery_dir / "recovery_summary.csv")
        section("Anchoring sensitivity with normalized recoveries")
        recovered_analysis = self.recovery_dir / "anchoring_analysis_recovered"
        table(recovered_analysis / "winner_change_rates.csv")
        table(recovered_analysis / "candidate_level_change_summary.csv")

        L = self.args.analysis_layer
        for cond in ("hidden", "visible"):
            mech = self.mech[cond]
            section(f"Probes ({cond}, layer {L}): activation decodability")
            table(
                mech / f"extra_offline_layer{L}" / "extra_probe_summary.csv",
                keep=["target", "test", "kind", "mean_auc", "mean_r2", "n_folds"],
                note="Compare activation_layer vs position_only rows.",
            )
            section(f"Residual beyond rank ({cond})")
            table(mech / f"extra_offline_layer{L}" / "residual_beyond_borda_summary.csv")
            section(f"Length control ({cond}): length-only baseline")
            table(mech / f"length_layer{L}" / "length_only_baselines.csv")
            section(f"Length control ({cond}): length-matched pooled AUC")
            table(
                mech / f"length_layer{L}" / "length_matched_control.csv",
                keep=["target", "length_stratum", "n_positive", "mean_auc"],
            )
            section(f"Residual beyond rank + length ({cond})")
            table(mech / f"length_layer{L}" / "residual_beyond_rank_with_length.csv")
            section(f"Direction reliability ({cond})")
            table(mech / f"dirpos_layer{L}" / "direction_reliability_control.csv")
            section(f"Position-matched decodability ({cond})")
            table(
                mech / f"dirpos_layer{L}" / "position_matched_control.csv",
                keep=["target", "shown_position", "n_positive", "mean_auc"],
            )
            section(f"Dimensionality / 1-D sufficiency ({cond})")
            table(mech / f"dim_layer{L}" / "score_dimensionality.csv")

        for cond in ("hidden", "visible"):
            out_dir = self.steer_dir[cond]
            if (out_dir / "causal_steering_summary.csv").exists():
                section(f"Causal steering summary ({cond})")
                table(out_dir / "causal_steering_summary.csv")
                section(f"Steering intensity signatures ({cond})")
                table(out_dir / "intensity" / "steering_intensity_summary.csv")
                section(f"Causal steering regressions ({cond})")
                table(out_dir / "causal_steering_regressions.csv")
                section(f"Causal steering spillovers ({cond})")
                table(out_dir / "causal_steering_middle_spillover_summary.csv")
                section(f"Steering baseline generation noise ({cond})")
                table(
                    out_dir / "causal_steering_baseline_noise.csv",
                    note=(
                        "Zero-intervention replicate-vs-primary differences at the "
                        "steering temperature: the per-ballot noise floor for the "
                        "steered deltas above."
                    ),
                )
                section(f"Causal steering quality control ({cond})")
                table(out_dir / "pipeline_quality_control.csv")

        legacy_caveat = ""
        qc_path = self.root / "vote_run_quality_control.csv"
        if qc_path.exists():
            qc = pd.read_csv(qc_path)
            if "legacy_complete_case_analysis" in qc and qc["legacy_complete_case_analysis"].astype(bool).any():
                omitted = int(qc["anchor_contexts_omitted_from_paired_runs"].max())
                legacy_caveat = (
                    f"- Legacy complete-case run: {omitted} anchor-invalid ballot contexts lacked "
                    "archived display mappings and were not attempted in the visible/placebo runs. "
                    "Paired estimates therefore condition on anchor ballot validity.\n"
                )
        lines.append(
            "\n## Caveats\n"
            + legacy_caveat
            + "- Primary estimates use strict-valid ballots. Normalized recovered "
            "ballots are reported only as a separately labeled sensitivity analysis.\n"
            "- Steering replays saved candidate order, ballot-field order, visibility, "
            "and strict-allocation settings from the source vote run.\n"
            "- Probe 'best cells' are exploratory model selection, not "
            "pre-registered confirmatory tests. The default fast layer is selected "
            "by architecture depth before outcomes are inspected.\n"
            "- Steering deltas are measured against the primary unsteered baseline "
            "(replicate 0). Additional zero-intervention baseline replicates "
            "quantify per-ballot generation noise directly "
            "(causal_steering_baseline_noise.csv), complementing the matched-norm "
            "random-direction controls and the hidden-placebo vote contrast.\n"
        )
        summary_path = self.root / "SUMMARY.md"
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[pipeline] Wrote {summary_path}", flush=True)

    # ---------- orchestration ----------
    def run(self) -> None:
        votes = self.root
        self.preflight()
        if self.args.preflight_only:
            print(f"[pipeline] Preflight passed. Provenance: {self.root / 'provenance_latest.json'}")
            return
        # Stage 1-3: elections.
        self.run_stage(
            "vote_hidden_anchor",
            self.vote_cmd(self.run_a, hidden=True, seed=self.args.anchor_seed, paired=False),
            [self.run_a / "direct_votes.csv", self.run_a / "candidates.csv"],
        )
        self.run_stage(
            "vote_visible",
            self.vote_cmd(self.run_v, hidden=False, seed=self.args.treatment_seed, paired=True),
            [self.run_v / "direct_votes.csv"],
        )
        self.run_stage(
            "vote_hidden_placebo",
            self.vote_cmd(
                self.run_b,
                hidden=True,
                seed=self.args.treatment_seed,
                paired=True,
                reuse_paired_inputs=True,
            ),
            [self.run_b / "direct_votes.csv"],
        )
        if self.args.dry_run:
            print("[pipeline] Dry run complete; no stages executed.")
            return
        self.validate_vote_runs()
        # Stage 4: anchoring with placebo correction.
        self.run_stage(
            "anchoring_analysis",
            [
                sys.executable, "-u", "level25_anchoring_placebo_analysis.py",
                "--anchor-dir", str(self.run_a),
                "--visible-dir", str(self.run_v),
                "--placebo-dir", str(self.run_b),
                "--evaluator-model", self.args.model,
                "--output-dir", str(self.anchoring_dir),
            ],
            [self.anchoring_dir / "winner_change_rates.csv",
             self.anchoring_dir / "corrected_effect_paired.csv"],
        )
        self.run_stage(
            "anchoring_voter_level",
            [
                sys.executable, "-u", "analyze_reference_anchoring.py",
                "--hidden-dir", str(self.run_a),
                "--visible-dir", str(self.run_v),
                "--output-dir", str(self.anchoring_detail_dir),
            ],
            [self.anchoring_detail_dir / "anchoring_candidate_pairs.csv",
             self.anchoring_detail_dir / "anchoring_ballot_change_summary.csv"],
        )
        # Same detailed voter-level analysis on the seed-only contrast (A vs B),
        # so every treatment table above has a matching sampling-noise floor.
        self.run_stage(
            "anchoring_voter_level_placebo",
            [
                sys.executable, "-u", "analyze_reference_anchoring.py",
                "--hidden-dir", str(self.run_a),
                "--visible-dir", str(self.run_b),
                "--allow-second-hidden",
                "--comparison-label", "hidden_anchor_vs_hidden_placebo",
                "--output-dir", str(self.anchoring_floor_dir),
            ],
            [self.anchoring_floor_dir / "anchoring_candidate_pairs.csv",
             self.anchoring_floor_dir / "anchoring_ballot_change_summary.csv"],
        )
        self.run_stage(
            "allocation_recovery_sensitivity",
            [
                sys.executable, "-u", "level25_recover_allocation_ballots.py",
                "--runs", str(self.run_a), str(self.run_v), str(self.run_b),
                "--output-dir", str(self.recovery_dir),
            ],
            [self.recovery_dir / "recovery_summary.csv",
             self.recovery_dir / self.run_a.name / "direct_votes.csv",
             self.recovery_dir / self.run_v.name / "direct_votes.csv",
             self.recovery_dir / self.run_b.name / "direct_votes.csv"],
        )
        recovered_anchor = self.recovery_dir / self.run_a.name
        recovered_visible = self.recovery_dir / self.run_v.name
        recovered_placebo = self.recovery_dir / self.run_b.name
        recovered_analysis = self.recovery_dir / "anchoring_analysis_recovered"
        self.run_stage(
            "anchoring_recovered_sensitivity",
            [
                sys.executable, "-u", "level25_anchoring_placebo_analysis.py",
                "--anchor-dir", str(recovered_anchor),
                "--visible-dir", str(recovered_visible),
                "--placebo-dir", str(recovered_placebo),
                "--evaluator-model", self.args.model,
                "--output-dir", str(recovered_analysis),
            ],
            [recovered_analysis / "winner_change_rates.csv",
             recovered_analysis / "candidate_level_change_summary.csv",
             recovered_analysis / "ballot_level_change_summary.csv"],
        )
        # Stage 5-6: activation extraction (hidden and visible).
        self.run_stage(
            "mech_extract_hidden",
            self.extract_cmd(self.run_a, self.mech["hidden"]),
            [self.mech["hidden"] / "self_answer_activations.npz",
             self.mech["hidden"] / "self_answer_vote_rows_with_text.csv",
             self.mech["hidden"] / "activation_row_index.csv"],
        )
        self.validate_mech("hidden")
        self.run_stage(
            "mech_extract_visible",
            self.extract_cmd(self.run_v, self.mech["visible"]),
            [self.mech["visible"] / "self_answer_activations.npz",
             self.mech["visible"] / "self_answer_vote_rows_with_text.csv",
             self.mech["visible"] / "activation_row_index.csv"],
        )
        self.validate_mech("visible")
        # Stage 7: offline battery per condition (CPU; skipped in --gpu-only).
        if self.args.gpu_only:
            print("[pipeline] --gpu-only: skipping CPU offline battery (run it off-box).", flush=True)
        else:
            for cond in ("hidden", "visible"):
                for name, cmd, outputs in self.offline_cmds(self.mech[cond]):
                    self.run_stage(f"offline_{cond}_{name}", cmd, outputs)
        # Stage 8: steering.
        if self.args.steering_condition != "skip":
            conditions = (
                ["hidden", "visible"]
                if self.args.steering_condition == "both"
                else [self.args.steering_condition]
            )
            for cond in conditions:
                for name, cmd, outputs in self.steering_cmds(cond):
                    self.run_stage(name, cmd, outputs)
                self.validate_steering(cond)
        # Stage 9: summary + (gpu-only) package mech dirs for off-box analysis.
        if self.args.gpu_only:
            self.package_for_offline()
        else:
            self.write_summary()
        print(f"\n[pipeline] COMPLETE. Artifacts under {votes}", flush=True)
        if self.args.gpu_only:
            print(f"[pipeline] Download the mech_*.zip and run {self.root / 'run_offline_locally.sh'}", flush=True)
        else:
            print(f"[pipeline] Read {self.root / 'SUMMARY.md'} first.", flush=True)

    def package_for_offline(self) -> None:
        """Zip the mech dirs and emit a script that runs the CPU battery off-box."""
        import shutil

        L = self.args.analysis_layer
        packaged = []
        for cond in ("hidden", "visible"):
            mech = self.mech[cond]
            if mech.exists():
                archive = shutil.make_archive(str(mech), "zip", mech)
                packaged.append(Path(archive).name)
        # The exact offline commands, parameterised by the extracted mech dir.
        script_lines = [
            "#!/usr/bin/env bash",
            "# Run the CPU offline battery locally after downloading mech_hidden/ and",
            "# mech_visible/ (unzip the mech_*.zip from the GPU box first).",
            "set -euo pipefail",
            f'LAYER={L}',
            'for d in mech_hidden mech_visible; do',
            '  [ -d "$d" ] || { echo "missing $d (unzip mech_$d.zip)"; continue; }',
            '  python level25_extra_offline_analysis.py --mech-output-dir "$d" --layer "$LAYER" --output-dir "$d/extra_offline_layer$LAYER"',
            '  python level25_direction_position_controls.py --mech-output-dir "$d" --layer "$LAYER" --output-dir "$d/dirpos_layer$LAYER"',
            '  python level25_length_controls.py --mech-output-dir "$d" --layer "$LAYER" --output-dir "$d/length_layer$LAYER"',
            '  python level25_dimensionality_tests.py --mech-output-dir "$d" --layer "$LAYER" --output-dir "$d/dim_layer$LAYER"',
            '  python level25_cosine_reliability_sensitivity.py --mech-output-dir "$d" --layer "$LAYER" --repeats 50 --output-dir "$d/cosine_layer$LAYER"',
            'done',
            'echo "Offline battery complete. Anchoring result is already in anchoring_analysis/ from the GPU run."',
        ]
        script_path = self.root / "run_offline_locally.sh"
        script_path.write_text("\n".join(script_lines) + "\n", encoding="utf-8")
        manifest = {
            "mode": "gpu_only",
            "analysis_layer": L,
            "packaged_mech_archives": packaged,
            "gpu_stages_done": "vote runs, anchoring analysis, activation extraction, steering",
            "offline_pending": [
                "extra_offline_analysis", "direction_position_controls", "length_controls",
                "dimensionality_tests", "cosine_reliability_sensitivity",
            ],
            "download": packaged + ["anchoring_analysis/", "steering_*/", "vote_run_quality_control.csv"],
            "next": "unzip mech_*.zip locally, then: bash run_offline_locally.sh",
        }
        (self.root / "gpu_only_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"[pipeline] Packaged mech dirs: {packaged}", flush=True)


def main() -> None:
    Pipeline(parse_args()).run()


if __name__ == "__main__":
    main()
