# scribe_digest — scribe_digest (model mistralai/codestral-2508, 681 tokens)

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
| it01 | model | Projecting the 5 numerical features (past-only historical rates and session time gaps) into the FM's embedding space to compute pairwise interactions with the categorical IDs will allow the model to learn personalized a… | +0.0030 | -0.0005 | kept_champion | scored | pure_bpr_single 0.6388 (-0.0018 vs the full run); pure_bpr_ensemble 0.6407 (+0.0001 vs the full run); champion_equiv 0.6395 (-0.0011 vs the full run) | FM with projected numerical features: 0.6406 vs 0.6411, kept; early-stopped at epoch 5. |

Calibration: over 1 scored iterations your predicted gain exceeded the measured one by +0.0035 on average (predicted − measured); size the next prediction accordingly.
Totals: 1 iterations; promoted 0; attempts per direction: model 1; never attempted: feature, training, multitask, other.

# TASK
Write the synthesis (max 150 words) from the fact table above. Numbers only from the table; no causal claims of your
own; no recommendations. Output only the synthesis.

## assistant (response)

The only direction attempted so far is model. It01 tried projecting numerical features into the FM's embedding space to learn personalized interactions with categorical IDs. This resulted in a measured Δ of -0.0005 vs the champion, leading to keeping the champion. The FM with projected numerical features scored 0.6406, slightly below the champion's 0.6411, and was early-stopped at epoch 5. No other directions have been attempted.
