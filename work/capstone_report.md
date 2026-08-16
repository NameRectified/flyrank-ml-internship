# Capstone Report — Content opportunity ranking (CTR / engagement fix queue)

- **Author:** Balaji (intern)
- **Lane:** ML — content opportunity scoring
- **Repo:** https://github.com/NameRectified/flyrank-ml-internship
- **Date:** 2026-08-16

## 1. Problem framing

A FlyRank content reviewer has limited time and a pool of tens of thousands of pages.
The decision this supports: **which pages should a reviewer open first, and for what kind
of fix** (CTR: title / meta / snippet / intent match; engagement: on-page structure /
depth / intent fulfillment).

- Unit of analysis: one **page** (content_hash_id) for one client, summarized over a
  feature window.
- Output: a **ranked, reason-coded queue** (score + action + reason codes).
- Cost of a wrong call:
  - False priority: review time spent on pages with low volume or a gap that is mostly noise.
  - Missed opportunity: a page that could have improved never gets looked at.

Why data/ML helps: one rule like "CTR below tier" does not fit every page — CTR gaps differ
by content type and engagement gaps by intent, so a score can rank the whole queue instead of
flagging one slice. The question is whether a transparent rule ranks that queue as well as a
trained model on **clients the model never saw**.

## 2. Data safety

Used (gated HF release, two tables):

- `fact_content_daily_performance`: daily impressions, clicks, average position, sessions,
  engaged sessions per page.
- `dim_content`: content type, main intent, created / updated dates.

Date windows: feature window **2026-01-01 to 2026-02-28**; outcome window **March 2026**;
decision moment **2026-03-01**.

Deliberately excluded and why:

- `trend_direction`, `trend_pct`: they answer a different question (trend decline), and are
  label-derived — never features.
- Measurement flags (`gsc_data_available`, `ga4_data_available`): used only to filter rows.
- `content_created_date`, `content_updated_date`: **audit columns only, never features**
  (we check for "created after the decision moment" artifacts).
- Pages under 100 feature-window impressions are not loaded; the queue floor of 500
  impressions keeps low-volume CTR noise out (a stated trade-off).

Leakage checks done (w03_feature_leakage_check): every feature ends before March 1; no
product flags exist; an earlier label that also required the feature-window gap was a leak and
was removed (9,858 pages would differ under it); tier medians recomputed on training data only
changed precision by almost nothing (mild but documented).

Nothing client-identifying appears in `work/`: only pseudonymized hashes, no client names,
URLs, or raw queries.

## 3. Baseline

A transparent rule, the same one the action playbook uses:

    score = has_volume * max(tier_ctr_gap, 0) * impressions_fw
    has_volume = impressions_fw >= 500

It ranks pages by exposure times how far CTR sits below the tier-median CTR, with a 500-
impression floor. It is fair because it runs on the **same pages, same label, same metric**
(precision@k) as the models.

Numbers on the pre-registered client holdout (seed 42, 0.1pp definition, base 15.9%):

- rule precision@10 **30%**, precision@50 **44%** (vs LR 50% / 32%).

At the validated stricter definition (seed 2026, 0.30pp, base 18.6%) the rule sits near its
own base-rate floor: precision@50 **22%** (vs LR **92%**).

## 4. Model / analysis

Logistic regression (w05 pipeline): scaled numeric features + one-hot encoded categories,
balanced class weights, seed 42. It fits the lane because it ranks the whole queue with a
single score and exposes per-feature coefficients for explanation.

Target (one sentence): **a page is "below tier" when its March CTR is more than X percentage
points below the weighted CTR median of its position tier**, with X = 0.1pp at the committed
definition and X = 0.30pp at the validated definition (0.30pp chosen on train only by a
coverage rule: strictest cutoff still leaving >=10% of pages below tier).

Features (all knowable at the decision moment):

- numeric: `log_impressions_fw`, `ctr_fw`, `avg_pos_fw`, `pos_volatility_fw`,
  `engagement_rate_fw`, `log_sessions_fw`, `tier_ctr_gap`
- categorical: `content_type`, `main_intent`, `position_tier`

A random forest did not generalize to held-out clients and its precision was not stable run
to run in our environment, so it is reported only qualitatively.

## 5. Evaluation

Split: **client holdout** (GroupShuffleSplit, whole clients held out) so the queue is scored
for clients the model never saw; a random split is shown only as the "BEFORE" trap (pages from
the same client leak across both sides and the models look near-perfect). Pre-registered
virgin seeds: **42** (committed 0.1pp) and **2026** (validated 0.30pp).

Metrics: precision@k vs the test base rate, on the same split. Model vs baseline:

| 0.1pp (seed 42, base 15.9%) | p@10 | p@50 |
|---|---|---|
| rule | 30% | 44% |
| logistic regression | 50% | 32% |

| 0.30pp (seed 2026, base 18.6%) | p@10 | p@50 | lift50 |
|---|---|---|---|
| rule | 30% | 22% | x1.18 |
| logistic regression | 80% | 92% | x4.96 |

Error analysis: the rule at 0.1pp fails on high-volume page-1 pages with a real gap that did
not persist into March (the "unusual pattern" false negatives); LR transfers best to the very
top of the queue where `tier_ctr_gap` dominates (coefficient ~4.1). At 0.30pp the rule stops
separating pages because exposure-times-gap can't distinguish "far behind tier median" pages.
The comparison is **not stable across splits or label definitions** (xp-06: seeds 42–45
sweep); the 0.30pp LR win is the one pre-registered confirmatory result that survived audits
(precision holds to k=200, 6% created-after-decision, spread over 2 content types and 6 of 10
clients).

## 6. Interpretation

- `tier_ctr_gap` dominates LR's coefficients — a page's distance below its position tier's
  CTR is the strongest single signal, which is the same signal the rule leans on.
- Higher positions capture more clicks per impression (tier gradient 0.41% top_3 down to
  0.05% deep) — descriptive, not proof that editing lifts CTR.
- Surprise / negative result: at the strict 0.30pp definition the transparent rule collapses
  to its base-rate floor while LR clearly separates; conversely, at the loose 0.1pp cut the
  rule holds the wider cut. Neither model is "always better"; the choice depends on the
  definition.
- A later fresh split (xp-08, seed 2027) made every model look perfect (100% precision@k);
  the decisive test showed this is a client-mix separation effect on that split, **not a
  claimed win** — no model beat the validated LR, so nothing was adopted.

## 7. Recommendation

Ranked queue for a reviewer, sorted by score, reason-coded:

- `ctr_opportunity` — visible page below its tier CTR. **The validated signal.**
- `engagement_gap` — sessions but engagement below `eng_target` (weighted median engagement
  by content type × intent). Directional context.
- `refresh_decay` — old / stale and still visible. Context from the paper's freshness findings.

Archetype → action: visible_ctr_underperformer → review_snippet_ctr; visible_ctr_striking →
improve_relevance_striking; visible_ctr_stale → review_snippet_and_refresh;
visible_engagement_weak → review_onpage_engagement; mature_visible → refresh_mature_page;
low_visibility → monitor.

**Validated alternative (decision support):** teams that adopt the stricter 0.30pp "below
tier" definition should rank with the logistic regression (precision@50 92% vs the rule's
22%, ~5x base rate, on a pre-registered fresh client holdout). Teams that keep the committed
0.1pp definition keep the transparent rule as the baseline (it holds the wider cut there).

Confidence and limits: decision support for one snapshot, portfolio-level tier medians, a
directional (not validated) engagement arm, a partly noisy label (no volume floor on March),
and a single fresh split for the 0.30pp win. Never automated: editing, publishing, deleting,
merging, or refreshing without a human review. No causal claims.

## 8. Reproducibility

From a fresh clone:

1. `pip install -r requirements.txt` (duckdb, pandas, numpy, scikit-learn, matplotlib; the
   HF token comes from `~/.env` via `HF_TOKEN`, or a prompt).
2. Run `work/notebooks/capstone.ipynb` top to bottom (Colab badge at the top works; the
   data release is gated by a HF read token).
3. Milestone and experiment notebooks: `w01` … `w07`, plus `xp-06` (cutoff sweep, seeds
   42–45), `xp-07` (pre-registered 0.30pp confirmatory test, seed 2026), `xp-08` (stronger
   models, seed 2027) — full log in `work/experiments/README.md`.

Seeds: 42 (committed definition + model zoo), 43–45 (sweep robustness), 2026 (xp-07 virgin
holdout), 2027 (xp-08 virgin holdout). Data splits are client-grouped (GroupShuffleSplit).

Outputs: `work/outputs/baseline_action_score.csv`, `model_metrics.json`,
`action_playbook_queue.csv`, `pre_registered_cutoff.csv` (CSVs git-ignored by the CI leak
guard; the notebooks regenerate them from the warehouse).

---

> **Claims checklist:** observed / measured / directional / decision-support throughout.
> Every precision@k is reported next to its base rate. No causal claims. No
> client-identifying details. Numbers match a fresh re-run of the notebooks.