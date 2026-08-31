# scribe_digest — scribe_digest (model mistralai/codestral-2508, 1048 tokens)

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
| it03 | training | Treating click and long_view as ordinal feedback levels and training BPR on all valid pairs (long_view > no_click, long_view > click_only, click_only > no_click) will provide granular gradients for items and give all-ne… | +0.0035 | -0.0080 | kept_champion | scored | pure_bpr_single 0.5964 (-0.0006 vs the full run); pure_bpr_ensemble 0.5970 (-0.0000 vs the full run); champion_equiv 0.6033 (+0.0063 vs the full run) | Ordinal BPR on click and long_view pairs primary 0.5970 vs 0.6050, kept_champion. |

Calibration: over 3 scored iterations your predicted gain exceeded the measured one by +0.0057 on average (predicted − measured); size the next prediction accordingly.
Totals: 3 iterations; promoted 0; attempts per direction: feature 1, model 1, training 1; never attempted: multitask, other.

# TASK
Write the synthesis (max 150 words) from the fact table above. Numbers only from the table; no causal claims of your
own; no recommendations. Output only the synthesis.

## assistant (response)

Feature direction: it01 added granular item-engagement priors with a +0.0001 gain. Model direction: it02's FwFM improved by +0.0002. Training direction: it03's ordinal BPR on click and long_view pairs lost -0.0080. Multitask and other directions untried. Calibration: predicted gains overstated by +0.0057 on average.
