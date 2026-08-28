# scribe_lesson — scribe_lesson (model mock:claude-haiku-4-5, 313 tokens, estimated)

## system block 1

# ROLE: Scribe (lesson)

You write ONE sentence (at most 20 words) that records the lesson of a finished experiment for future
planning. You are given only harness-measured facts: the hypothesis, the status, the metrics, the
harness decision and the best score afterwards. Use only those facts. Never invent numbers, never
argue with the decision, never speculate beyond what the result supports. Name the direction and the
outcome so a reader of the one-line ledger learns something ("pairwise loss +0.004: aligning the loss
with the ranking metric pays off"; "LightGBM on id features crashed on memory; needs sparse encoding").
Output only the sentence — no quotes, no prefix, no newline.

## user

# Facts (measured by the harness)
HYPOTHESIS: Stronger L2 (1e-6 -> 1e-5) to regularise sparse id embeddings under temporal shift
CATEGORY: training
RESULT: {"status": "scored", "gauc": 0.6684909342470425, "ndcg5": 0.536514481883144, "primary": 0.6025027080650933, "runtime_s": 45.9, "error_excerpt": "", "vs_best": "+0.0010"}
DECISION (harness): promoted
BEST PRIMARY AFTER: 0.6025027080650933

# TASK
Write the one-sentence lesson (max 20 words) for the facts above. Output only the sentence.

## assistant (response)

Stronger L2 (1e-6 -> 1e-5) to regularise sparse: scored +0.0010 -> promoted
