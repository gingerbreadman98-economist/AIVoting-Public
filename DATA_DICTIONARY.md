# Data and terminology

## Ballot status

- **Exact-context replay**: saved candidates, self-answer, candidate order,
  ballot-field order, and prompt construction are replayed. This does not imply
  that the generated output satisfies the budget.
- **Natively budget-valid**: the output parses and its absolute allocation
  totals 100 cents without rescaling.
- **Normalized allocation**: parseable signed values are divided by their
  absolute total to produce a unit absolute budget.
- **Usable**: natively valid or normalized, with complete vote fields.
- **Unusable**: no complete finite allocation remains after retries.

## Behavioral targets

- **Ordinal information**: candidate ordering or pairwise comparison.
- **Polarity**: positive (help), zero (neutral), or negative (hurt) allocation.
- **Intensity**: allocation magnitude conditional on ordinal judgment and
  polarity.
- **Bipolar readout**: help and hurt are consistent with approximately opposite
  readouts of a shared decodable dimension. This does not imply cardinal
  utility or exact one-dimensional sufficiency.

## Principal files

- `direct_votes.csv`: one row per candidate within each usable ballot.
- `allocation_validation_diagnostics.csv`: one row per attempted ballot,
  including retries, native validity, normalization, and failure information.
- `self_answer_vote_labels.csv`: mechanistic row labels joined to ballot
  outcomes.
- `self_answer_activations.npz`: saved layer-16 activation arrays used by the
  offline probe scripts.
- `causal_steering_vote_rows.csv`: baseline and intervention candidate rows.
- `causal_steering_raw_outputs.csv`: one row per generation attempt, including
  validity and normalization indicators.

CSV is the canonical packaged tabular format. Duplicate JSONL copies from the
original runs are intentionally omitted.

