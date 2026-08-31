# scribe_digest — scribe_digest (model mistralai/codestral-2508, 909 tokens)

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
| it01 | feature | Providing the model with strictly past-only video and author historical click (valid play) and like rates as numerical features will inject granular item-engagement priors that differentiate between clickbait and high-p… | +0.0025 | +0.0001 | kept_champion | scored | pure_bpr_single 0.6030 (-0.0021 vs the full run); pure_bpr_ensemble 0.6051 (+0.0000 vs the full run); champion_equiv 0.6033 (-0.0018 vs the full run) | Primary 0.6051 vs 0.6050, kept; early-stopped at epoch 9. |
| it02 | model | Generalizing the Factorization Machine to a Field-weighted FM (FwFM) will allow the model to learn the importance of different field-pair interactions, upweighting critical crosses like user-video while discounting nois… | +0.0030 | -0.0002 | kept_champion | scored | pure_bpr_single 0.6030 (-0.0019 vs the full run); pure_bpr_ensemble 0.6049 (+0.0000 vs the full run); no_fwfm 0.6033 (-0.0016 vs the full run) | FwFM primary=0.6049 gauc=0.6717 ndcg5=0.5380 kept; early-stopped at epoch 9. |

Calibration: over 2 scored iterations your predicted gain exceeded the measured one by +0.0028 on average (predicted − measured); size the next prediction accordingly.
Totals: 2 iterations; promoted 0; attempts per direction: feature 1, model 1; never attempted: training, multitask, other.

# TASK
Write the synthesis (max 150 words) from the fact table above. Numbers only from the table; no causal claims of your
own; no recommendations. Output only the synthesis.

## assistant (response)

Two iterations have been scored. In iteration 1, adding granular item-engagement features improved the primary metric by +0.0001, but the pure_bpr_ensemble ablation showed no change. In iteration 2, replacing the Factorization Machine with a Field-weighted FM (FwFM) decreased the primary metric by -0.0002, but the pure_bpr_ensemble ablation showed no change. The FwFM primary metric was 0.6049, and it was kept. No iterations have been promoted. Attempts per direction: feature 1, model 1. Directions never attempted: training, multitask, other.
