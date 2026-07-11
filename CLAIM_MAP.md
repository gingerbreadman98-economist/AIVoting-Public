# Claim-to-artifact map

| Paper result | Effective unit | Primary artifact | Reproduction code |
|---|---:|---|---|
| Best-pick AUC 0.776; allocation-top AUC 0.829; allocation-z R2 0.242 | 100 prompt groups, 1,532 candidate rows | `data/primary_mechanistic/self_geometry_best_probe_layers.csv` | `level25_self_answer_activation_analysis.py` |
| Pairwise allocation preference AUC 0.787 | 100 prompt groups, 4,368 candidate pairs | `data/primary_mechanistic/extra_offline_layer16/pairwise_contrast_summary.csv` | `level25_extra_offline_analysis.py` |
| Residual intensity has no useful linear out-of-sample prediction | 100 prompt groups | `data/primary_mechanistic/extra_offline_layer16/residual_beyond_borda_summary.csv` | `level25_extra_offline_analysis.py` |
| Help/hurt score-sufficiency results | 5 prompt-grouped folds | `data/primary_mechanistic/dim_layer16/score_dimensionality.csv` | `level25_dimensionality_tests.py` |
| Position-matched probe controls | 4 position strata, grouped CV | `data/primary_mechanistic/dirpos_layer16/position_matched_control.csv` | `level25_direction_position_controls.py` |
| Reference-display winner changes and instability floor | 100 paired prompts | `data/reference_display/winner_change_rates.csv` and `corrected_effect_paired.csv` | `level25_anchoring_placebo_analysis.py` |
| Signed semantic steering coefficient +0.182 (SE 0.049) | 50 ballot clusters, 1,873 finite target outcomes | `data/steering/causal_steering_regressions.csv` | `analyze_causal_steering_regressions.py` |
| Baseline allocation noise 0.041 | 95 replicate ballots, 380 candidate pairs | `data/steering/causal_steering_baseline_noise.csv` | steering generation script |
| Steering validity: 213 native, 1,660 normalized, 227 unusable | 2,100 steered generations | `data/steering/causal_steering_raw_outputs.csv` | `level25_causal_activation_steering.py` |
| 14B behavioral validation table | 100 prompts | `data/behavioral_validation/method_summary.csv` | `level1_direct_vote_eval_vLLM.py` |
| Cross-model measurement-invariance warning | 400 attempts/model | `data/cross_model/*/direct_votes.csv` and diagnostics | `level25_self_answer_vote_vLLM.py` |

Row counts are repeated measurements unless the effective unit says otherwise.
Probe cross-validation is grouped by prompt; steering standard errors are
clustered by ballot.

