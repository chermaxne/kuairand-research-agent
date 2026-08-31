# scribe_digest — scribe_digest (model mistralai/codestral-2508, 769 tokens)

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
| it01 | model | Replacing the numpy FM with a PyTorch DeepFM and concatenating strictly past-only user/video historical rates as numerical features into the MLP will allow the model to learn high-order interactions and behavioral prior… | +0.0050 | n/a | failed | failed: [debugger abandoned: PyTorch (torch) is not installed in th… | — | PyTorch DeepFM failed due to missing torch module; no score. |
| it02 | training | Training the numpy FM with a within-user pairwise BPR loss directly aligns the optimization objective with the evaluation metrics (GAUC, nDCG@5), providing a stronger ranking signal than pointwise logloss. | +0.0029 | +0.0018 | promoted | scored | champion_equiv 0.6015 (-0.0018 vs the full run) | Pairwise BPR: 0.601470 vs 0.6032277176425208, promoted; early-stopped at epoch 9. |

Calibration: over 1 scored iterations your predicted gain exceeded the measured one by +0.0011 on average (predicted − measured); size the next prediction accordingly.
Totals: 2 iterations; promoted 1 (it02); attempts per direction: model 1, training 1; never attempted: feature, multitask, other.

# TASK
Write the synthesis (max 150 words) from the fact table above. Numbers only from the table; no causal claims of your
own; no recommendations. Output only the synthesis.

## assistant (response)

The only model change attempted was replacing the numpy FM with a PyTorch DeepFM (it01), which failed due to missing PyTorch. The only training change attempted was switching to pairwise BPR loss (it02), which succeeded with a measured gain of +0.0018 over the champion. No feature, multitask, or other directions have been attempted. Pairwise BPR was promoted, while PyTorch DeepFM failed. Calibration shows predicted gains overstated by +0.0011 on average.
