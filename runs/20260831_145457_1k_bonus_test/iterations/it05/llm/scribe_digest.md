# scribe_digest — scribe_digest (model mistralai/codestral-2508, 1433 tokens)

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
| it03 | feature | Adding user historical long_view rates and item/author auxiliary feedback rates (click, like) as past-only numerical features will provide DeepFM's MLP with rich interaction surfaces, allowing it to learn non-linear per… | +0.0030 | +0.0003 | promoted | scored | pure_bpr_single 0.6353 (-0.0139 vs the full run); pure_bpr_ensemble 0.6492 (+0.0000 vs the full run); champion_equiv 0.6395 (-0.0097 vs the full run) | Adding user historical long_view rates and item/author auxiliary feedback rates as past-only numerical features promote… |
| it04 | feature | Standardizing past-only numerical features will stabilize DeepFM's gradients against scale imbalances, adding missing user click/like rates will complete the behavioral priors, and within-user rank ensembling will optim… | +0.0025 | +0.0037 | promoted | scored | pure_bpr_single 0.6472 (-0.0056 vs the full run); pure_bpr_ensemble 0.6528 (-0.0000 vs the full run); champion_equiv 0.6395 (-0.0133 vs the full run) | DeepFM with standardized features and within-user rank ensembling promoted to new champion with primary score 0.6528. |
| it05 | feature | Adding past-only user-tab specific historical impression and positive rates as numerical features will give the DeepFM MLP a highly personalized, context-aware baseline for each user's generosity across different UI tab… | +0.0025 | +0.0035 | promoted | scored | pure_bpr_single 0.6479 (-0.0084 vs the full run); pure_bpr_ensemble 0.6562 (-0.0001 vs the full run); champion_equiv 0.6395 (-0.0168 vs the full run) | DeepFM MLP with past-only user-tab features: primary 0.6563 promoted. |
| it01 | model | Projecting the 5 numerical features (past-only historical rates and session time gaps) into the FM's embedding space to compute pairwise interactions with the categorical IDs will allow the model to learn personalized a… | +0.0030 | -0.0005 | kept_champion | scored | pure_bpr_single 0.6388 (-0.0018 vs the full run); pure_bpr_ensemble 0.6407 (+0.0001 vs the full run); champion_equiv 0.6395 (-0.0011 vs the full run) | FM with projected numerical features: 0.6406 vs 0.6411, kept; early-stopped at epoch 5. |
| it02 | model | Extending the FM to a DeepFM by adding a 1-layer MLP over the concatenated embeddings and numerical features will allow the model to learn arbitrary high-order feature interactions, providing a stronger personalization… | +0.0025 | +0.0078 | promoted | scored | pure_bpr_single 0.6347 (-0.0142 vs the full run); pure_bpr_ensemble 0.6489 (+0.0000 vs the full run); champion_equiv 0.6395 (-0.0094 vs the full run) | DeepFM primary=0.6489 gauc=0.6762 ndcg5=0.6215, promoted. |

Calibration: over 5 scored iterations your predicted gain exceeded the measured one by -0.0002 on average (predicted − measured); size the next prediction accordingly.
Totals: 5 iterations; promoted 4 (it02, it03, it04, it05); attempts per direction: feature 3, model 2; never attempted: training, multitask, other.

# TASK
Write the synthesis (max 150 words) from the fact table above. Numbers only from the table; no causal claims of your
own; no recommendations. Output only the synthesis.

## assistant (response)

Feature direction: 3 iterations, 3 promotions. Adding user historical long_view rates and item/author auxiliary feedback rates (it03) and past-only user-tab specific historical impression and positive rates (it05) promoted primary scores to 0.6492 and 0.6563, respectively. Standardizing past-only numerical features and adding missing user click/like rates (it04) promoted primary score to 0.6528. Model direction: 2 iterations, 1 promotion. Extending FM to DeepFM (it02) promoted primary score to 0.6489. Projecting numerical features into FM's embedding space (it01) kept primary score at 0.6406. Training, multitask, and other directions untried.
