# Capstone model experiments — why and what we tried

This file is the honest record of the experiment round: why we ran it, what we
decided in advance, what the guardrails are, and what each run showed. Append to
the log at the bottom; never rewrite a past verdict.

## Why we are experimenting

The deployed paper's honest result is: rule-based baseline 44% vs logistic
regression 32% at precision@50 on held-out clients, against a 15.9% base rate.
Read plainly, the numbers say a simple, transparent rule is enough and ML adds
little. That is a legitimate finding, but it deserves one honest test before we
settle it, for two reasons:

1. **The label and the rule share the same signal.** The label
   `below_tier_outcome` is defined as "CTR in the outcome window more than 0.1pp
   below the position-tier median", and the baseline rule is exactly
   `has_volume * max(tier_ctr_gap, 0) * impressions`. A model trained on
   `tier_ctr_gap` is reconstructing the rule (its top logistic coefficient is
   `tier_ctr_gap` at 4.1), so it cannot beat the rule by construction. The only
   honest way ML can win is with signal the rule does not use.
2. **The evaluation is noisy at the top.** At a 15.9% base rate, precision@10 is
   measured on about 1.6 expected hits, so two-point tables wobble run to run.

So this round tests: can a model — using stronger techniques (gradient
boosting, blends) on the SAME proven-safe features the main line uses, or new
features knowable at the decision moment — beat the transparent rule on clients
it never saw? The first two runs leaked (the dim_content "enrichment" columns
are July-snapshot values = future data). xp-03 narrowed to technique only
(honest negative); xp-04 hit a config artifact in XGBoost/LightGBM; xp-05 added
safe momentum/engagement features whose apparent win proved to be a page-age
cohort artifact (details below). Every run reached the same answer: no model
beats the rule on the pages that existed at the decision moment.

## Pre-registered win criterion (decided before any run)

A model counts as a win only if it beats the rule-based baseline at **both**
precision@10 and precision@50 on the client-holdout split, by a clear margin
above the ~2x base-rate floor, and that result must survive:

- the same label (`below_tier_outcome`), same split (client holdout,
  GroupShuffleSplit, test_size 0.2, seed 42), same cutoffs;
- hyperparameters chosen by client-grouped cross-validation inside the train
  split only, never by looking at the holdout;
- the precision@k curve (k = 10..200), not a single lucky cutoff.

Anything less is an honest negative result, and the paper keeps the framing
fix instead (drop the unstable random-forest row from the table, lead with the
~2.8x lift over base rate, correct the abstract).

## Guardrails

- **No leakage.** All features must be knowable at the decision moment
  (March 1, 2026). `fact_content_query_90d` is excluded: its fixed 90-day window
  covers the recent months past March 1, so its columns are future data for this
  problem. xp-01/xp-02 showed the same problem in `dim_content`: updated date,
  keyword context, and even word count are July-snapshot values (pages rewritten
  after March changed them post-decision). From xp-03 on, **no dim_content
  enrichment is used in modeling at all** — only the proven-safe w05 feature
  set, and the leak is documented in a diagnostic cell for the record.
- **No tuning on the holdout.** A nested client-holdout inside train decides all
  hyperparameters and blend weights.
- **Report everyone.** The summary table lists every method tried on the same
  cutoffs; no dropping of losers.
- **Random forest stays qualitative** in the paper (its precision was not stable
  run to run in our environment).
- **Same comparison base.** The experiment reuses the w05 load query verbatim so
  every method sees the same 120,258 pages and the same split.

## Experiment log

| Run | Date | What changed | Rule @10/@50 | Best model (@10/@50) | Verdict |
|---|---|---|---|---|---|
| xp-01 | 2026-08-09 | First run. Added dim_content richness (word_count, cpc, competition_level, created + updated dates) and an exposure-by-gap interaction. | 30% / 44% | HistGB "100% / 88%" | **FAIL — leakage.** `content_updated_date` is a July-snapshot value, so post-decision updates (April–June) are future data that track the label; `competition_level`/`cpc` are suspected the same (keyword context from the overlapping 90-day window). The inflated numbers were discarded. A secondary bug (pandas `merge` reordered rows) silently drifted the split — `lr_cur` showed 38% @50 vs 32% in w05. |
| xp-02 | 2026-08-09 | Removed updated-date + keyword-context features (the xp-01 leak); kept word_count/char_count/created-date; fixed split order so the rule reproduces 30/44 exactly. | 30% / 44% | HistGB 100% / 80% | **FAIL / inconclusive.** HistGB still hits 80% @50 and stays ~80% to k=200 despite removing the obvious leak. Suspect the remaining dim_content columns (word_count/char_count are July-snapshot: pages rewritten after March changed length post-decision, which tracks the label) or an overfit/class-shift artifact. xp-03 tests technique-only on the proven-safe w05 features and checks a second split seed. |
| xp-03 | 2026-08-09 | Technique-only on the proven-safe w05 feature set (no dim_content enrichment; leak documented in a diagnostic cell: 78.2% of pages updated after decision, word_count median 2738 vs 3221, 7.1% created after decision). Two client-holdout seeds (42 + 43). | 30% / 44% (seed 42, sanity) | HistGB 50% / 20% — no method beats the rule at both cuts (lr 50/32, blend_rule_rf_cur 50/36, rf 20/4, et 30/14; seed 43 is a 68.8% base-rate ceiling) | FAIL — honest negative. |
| xp-04 | 2026-08-09 | Same safe features; added XGBoost/LightGBM; full tracebacks + seed-43 zoo with relative lift. | 30% / 44% (exact repro) | histgb 40% / 22%; XGB 0/0, LGBM 0/2 — config artifact | FAIL — no winner. XGB/LGBM collapsed because scale_pos_weight (< 1 on the majority class) + AUC early stopping is a degenerate pairing; rule and lr_cur repro unchanged. |
| xp-05 | 2026-08-09 | Added decision-time-safe momentum/engagement features: impression_trend (Feb/Jan), engagement_gap, days_since_created, has_jan_imp/has_created_date. Fixed XGB/LGBM (logloss, no scale_pos_weight); added CatBoost/MLP/kNN/SGD/GaussianNB + stacking meta. Seeds 42-45. | 30% / 44% (exact repro); lr_cur 50/32 exact | HistGB/XGB/LGBM/CatBoost 100% / 100% on seed 42 (AUPRC 35-41%) | FAIL — artifact (see note below). |

### Why the xp-05 "100% / 100%" is not a win (audit, 2026-08-09)

The headline result was re-examined before touching the paper. Findings:

- **Feature ablation** (HistGB, seed 42): the entire win comes from
  `days_since_created`. Removing it drops p@50 to ~70%; removing
  `impression_trend` or `engagement_gap` changes nothing. Removing all new
  features returns to the xp-03/04 level (~28% p@50).
- **Top-100 anatomy**: 82% have `days_since_created == 0` (= created on/after
  March 1), 97% have no January impressions, 79% zero sessions, 100% are
  `keyword article` at page-1/top-3 positions, from only 4 of the 10 test
  clients. The model was isolating one niche: brand-new pages in top positions
  with zero engagement.
- **Decisive test** — restrict evaluation to pages that existed on the decision
  date (exclude the ~10% created after March 1, which are not actionable at
  decision time): HistGB p@50 falls to 58%, LR to 38%, rule stays 44%. Restrict
  further to pages with ≥ 1 month of history: HistGB collapses to 2% p@50, LR
  20%, rule 38%.

Conclusion: on the population a content team could actually rank on March 1,
every model is at or below the transparent rule. The xp-05 notebook's
auto-printed PASS reflected the inflated full-test numbers; the corrected
verdict is FAIL, and the pre-registered framing fix stands.

## Outcome for the paper

xp-01 and xp-02 leaked. xp-03 (technique-only) and xp-04 (config artifact) were
honest negatives. xp-05 looked like a win but the audit showed it was a
page-age cohort artifact — on the decision-time-actionable pages every model is
at or below the rule. The paper keeps the framing fix: lead with the ~2.8x lift
over the 15.9% base rate, correct the abstract, and drop the unstable
random-forest row.
