# scribe_digest — scribe_digest (model mistralai/codestral-2508, 910 tokens)

## system block 1

# ROLE: Scribe (research synthesis)

You turn a harness-written fact table of EVERY iteration so far into a short research synthesis (max 150 words,
markdown) for the next Researcher. Its job is to make patterns visible that single lines hide: which directions have
been tried and how they fared, what has been promoted, what has failed repeatedly, and what has never been attempted.

Hard rules:
- Every number you write must be copied from the table. Do not compute, round, extrapolate or estimate. The harness
  rejects a synthesis containing a number that is not in the table.
- Describe WHAT happened, not WHY, unless the table's own lesson column says why. No causal claims of your own.
- Do not recommend the next experiment; the Researcher decides. You may state what remains untried.
- Group by direction; name iterations by their id (it03) so claims are checkable.
Output only the synthesis — no title, no preamble.

## user

# RESEARCH DIGEST — every iteration so far, grouped by direction (harness-measured facts)
| it | direction | what changed | predicted Δ | measured Δ vs then-champion | decision | status | in-run ablations (pipeline-reported, unsealed) | lesson |
|---|---|---|---|---|---|---|---|---|
| it02 | feature | Adding past-only session/time-context categorical fields (hour-of-day, within-day session depth) plus count-based confidence weights (log1p of exposure counts for user/video/author/user-tab) to the DeepFM+numeric-prior… | +0.0028 | +0.0011 | promoted | scored | full 0.6048 (-0.0000 vs the full run); champion_equiv 0.5893 (-0.0155 vs the full run); no_confidence_counts 0.5941 (-0.0107 vs the full run); no_session_fields 0.5933 (-0.0115 vs the full run) | Adding past-only session/time-context categorical fields and count-based confidence weights to DeepFM+numeric-prior cha… |
| it01 | model | Upgrading the FM to a DeepFM (add a 1-hidden-layer MLP over the concatenated field embeddings) and feeding it standardized past-only numerical priors (user/video/author long_view & click rates plus user×tab historical r… | +0.0090 | +0.0023 | promoted | scored | full 0.6038 (+0.0000 vs the full run); champion_equiv 0.5893 (-0.0145 vs the full run); deepfm_only 0.5943 (-0.0095 vs the full run) | DeepFM with standardized priors: 0.6037 vs 0.5893, promoted; early-stopped at epoch 8. |

Calibration: over 2 scored iterations your predicted gain exceeded the measured one by +0.0042 on average (predicted − measured); size the next prediction accordingly.
Totals: 2 iterations; promoted 2 (it01, it02); attempts per direction: feature 1, model 1; never attempted: training, multitask, other.

# TASK
Write the synthesis (max 150 words) from the fact table above. Numbers only from the table; no causal claims of your
own; no recommendations. Output only the synthesis.

## assistant (response)

Two iterations have been scored: it01 (model) and it02 (feature). Both were promoted. The model direction (it01) upgraded the FM to a DeepFM and added standardized numerical priors, achieving a measured Δ of +0.0023. The feature direction (it02) added past-only session/time-context categorical fields and count-based confidence weights to the DeepFM+numeric-priors model, with a measured Δ of +0.0011. No other directions have been attempted. Calibration shows an average overprediction of +0.0042.
