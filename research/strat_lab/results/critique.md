# Intraday Strategy Study: Head-of-Research Review

Data: NIFTY-100 5-minute bars, 60 sessions (2026-07-23 to 2026-10-16). TRAIN has 37 sessions and TEST has 23. TEST has 23 sessions, not 24, because 2026-09-14 (a Monday) is missing from the cache and 2026-10-02 was a holiday. All figures are in rupees on Rs 10L capital, net of Indian intraday charges and 1bp/side slippage, unless stated otherwise.

What I checked myself:
- Re-ran `index_trend` on TRAIN and TEST. It matches to the rupee.
- Rebuilt every strategy's daily P&L from the results CSVs. TRAIN and TEST t-stats match the reported ones.
- Broke the baseline's loss into charges, slippage, market beta and stock selection.
- Ran three minimal changes to the baseline, B1–B3. I fixed them before running and ran each once on TRAIN and once on TEST, with no tuning.

**Bottom line:** none of the 7 strategies or 14 portfolios has shown an edge. The study did find out why the live rules lose, and two of those causes can be fixed in code today: turnover and gross long exposure.

---

## 1. Why the baseline loses, ranked

Over all 60 days the baseline (`app_current`) lost **-Rs 1,85,566**: 781 trades, 38% win rate, profit factor (PF) 0.71, daily t-stat -2.59. I split that loss into parts:

| Component | Rs | Share of loss | How it was measured |
|---|---|---|---|
| Charges | -91,505 | 49% | 781 round trips × about Rs 117. That is 0.059% of a Rs 2L position. |
| Slippage | -31,013 | 17% | 1bp per side × 781 trades × about Rs 2L, so Rs 40 per trade. |
| Market beta | -64,800 (±21,600 at 95%) | 35% | I regressed daily P&L on NIFTY's 10:00→15:15 return: about Rs 15,295 per 1% NIFTY move (SE 2,601, R² 0.37). That is like holding about Rs 15L of NIFTY long every day. NIFTY's 10:00→15:15 moves added up to -4.24% over the 60 days. |
| Stock selection (what is left) | **about +1,750** | **0%** | Gross -94,061, minus the beta and slippage parts. |

The causes, in order of size:

1. **A signal with no edge, traded about 13 times a day.** Costs (charges plus slippage) are Rs 1,22,518, which is 66% of the loss.
   - Stock selection adds about zero. That fits the forward returns after entry: -0.013% at 30 minutes, -0.033% at 60 minutes, +0.002% to the close.
   - Across the whole universe, stocks up 1–5% at 09:55 returned -0.04% to 12:00 and -0.01% to the close.
   - Charges alone run at Rs 1,525 a day, which is 0.15% of capital a day, or about 38% a year.
2. **Built-in long market beta in a market that drifted down.** Daily P&L correlates +0.43 with NIFTY over all 60 days (+0.55 on TEST), and beta accounts for about 35% of the loss.
   - By NIFTY's day: -Rs 1,41,273 on 17 down days, -Rs 1,03,845 on 29 flat days, +Rs 59,553 on 14 up days.
3. **Everything is bought in a burst at 10:00.** 512 of 781 entries (66%) came in the 10:00 hour, and 337 (43%) in the first three bars (10:00–10:10).
   - That hour lost Rs 1,53,191, which is 83% of the total.
   - Up to 10 positions of Rs 2L each means up to Rs 20L (2x capital) of correlated longs within minutes. This is the mechanism behind cause 2.
4. **The exits assume a payoff that never happens.** The 2.5% target needs a move the stocks do not make: the median best move after entry is +0.52%, and only 3.7% of trades ever touched +2.5%.
   - Only 24 of 781 exits (3%) hit the target. 58% were time exits.
   - The 2.08:1 reward-to-risk on paper became avg win Rs 1,516 vs avg loss Rs 1,319 (1.15x) at a 38% win rate.
5. **The daily loss limit fires on market-down days.** 36 trades were flattened by the -2% limit on 5 days, for -Rs 33,716. The breaker works as designed. It fires because of causes 2 and 3.

What is *not* the cause: per-trade sizing. It is now enforced in code and is correct.

## 2. Scoreboard

All runs at 1bp/side slippage. Max DD is measured on end-of-day equity. "Configs" means how many settings were tried on TRAIN.

**Baseline and my three fixed variants (each run once):**

| Book | TRAIN net | Trades | PF | Max DD % | t | TEST net | Trades | PF | Max DD % | t | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (`app_current`) | -1,03,487 | 486 | 0.73 | 10.95 | -1.96 | -82,079 | 295 | 0.68 | 9.60 | -1.67 | Loses |
| B1: max 3 trades/day, biggest gainers first | -15,337 | 108 | 0.83 | 2.38 | -0.59 | -29,930 | 68 | 0.57 | 4.19 | -1.50 | Loses less; the change is mechanical, not an edge |
| B2: longs only if NIFTY is above its day open and previous close | -91,020 | 187 | 0.46 | 9.20 | -2.31 | -3,060 | 196 | 0.98 | 3.38 | -0.08 | Inconsistent (noise) |
| B3: B1 + B2 | -21,465 | 45 | 0.55 | 2.37 | -1.09 | -435 | 45 | 0.99 | 1.48 | -0.03 | Inconsistent |

**The seven strategies:**

| Strategy | Configs | TRAIN net | Trades | PF | Max DD % | t | TEST net | Trades | PF | Max DD % | t | TEST at 2bp | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regime_momentum | 13 | +3,840 | 38 | 1.27 | 0.74 | 0.44 | -1,929 | 50 | 0.93 | 0.84 | -0.26 | -5,171 | no_edge |
| orb | 11 | +39,868 | 183 | 1.30 | 1.46 | 1.23 | -41,681 | 134 | 0.74 | 5.85 | -1.34 | -46,304 | no_edge |
| vwap_reversion | 16 | +1,078 | 155 | 1.02 | 0.94 | 0.08 | -17,063 | 127 | 0.70 | 1.60 | -1.14 | -21,970 | no_edge |
| gap | 14 | +86,587 | 159 | 1.97 | 0.95 | 2.66 | -25,066 | 101 | 0.76 | 3.17 | -1.21 | -29,098 | no_edge |
| rs_neutral | 19 | +56,445 | 132 | 1.72 | 1.35 | 2.28 | -8,651 | 92 | 0.90 | 1.60 | -0.46 | -12,119 | no_edge |
| index_trend | 12 | +37,729 | 95 | 1.68 | 1.52 | 1.36 | **+13,661** | 75 | 1.25 | 1.84 | 0.58 | +11,089 | marginal (unproven) |
| low_freq_breakout | 21 | +60,969 | 54 | 3.60 | 0.70 | 2.78 | -30,803 | 39 | 0.40 | 3.17 | -2.28 | -32,168 | no_edge |

**The 14 portfolios.** "Slices" gives each strategy its own share of capital. "Shared" runs them all in one Rs 10L account at full size.

| Portfolio | TRAIN net | Trades | PF | Max DD % | t | TEST net | Trades | PF | Max DD % | t |
|---|---|---|---|---|---|---|---|---|---|---|
| EW_all_slices | +7,096 | 809 | 1.09 | 0.57 | 0.66 | -40,860 | 612 | 0.54 | 4.23 | -4.56 |
| EW_all_shared | +1,13,006 | 423 | 1.43 | 1.65 | 2.44 | -90,498 | 279 | 0.69 | 10.57 | -2.06 |
| TOP_T_slices | +56,510 | 345 | 1.84 | 0.32 | 3.25 | -28,303 | 232 | 0.67 | 3.07 | -3.04 |
| TOP_T_shared (picked by the rule fixed in advance) | +1,71,258 | 307 | 1.99 | 0.80 | 3.71 | -64,492 | 210 | 0.71 | 7.31 | -2.24 |
| RG_beta_slices | +756 | 704 | 1.01 | 0.79 | 0.09 | -31,691 | 520 | 0.57 | 3.35 | -3.50 |
| RG_beta_shared | +1,01,137 | 413 | 1.40 | 1.63 | 2.36 | -35,450 | 291 | 0.87 | 8.14 | -0.71 |
| RG_trend_slices | -3,019 | 406 | 0.93 | 0.55 | -0.44 | -17,668 | 309 | 0.63 | 2.02 | -2.46 |
| RG_trend_shared | +54,888 | 328 | 1.26 | 1.54 | 1.31 | -11,758 | 219 | 0.94 | 3.98 | -0.36 |
| RG_fit_slices | +18,325 | 591 | 1.33 | 0.28 | 1.75 | -24,428 | 344 | 0.56 | 2.60 | -3.91 |
| RG_fit_shared | +1,56,054 | 372 | 1.68 | 1.07 | 2.98 | -61,041 | 231 | 0.74 | 6.78 | -2.04 |
| PERF5_slices | +13,109 | 566 | 1.26 | 0.29 | 1.57 | -14,961 | 193 | 0.49 | 1.71 | -2.41 |
| PERF5_shared | +1,48,671 | 369 | 1.74 | 1.28 | 3.32 | -53,811 | 167 | 0.67 | 6.74 | -1.53 |
| PERF10_slices | +7,355 | 605 | 1.13 | 0.38 | 0.81 | -11,053 | 150 | 0.48 | 1.19 | -3.00 |
| PERF10_shared | +1,13,862 | 377 | 1.50 | 1.51 | 2.54 | -34,942 | 132 | 0.70 | 5.21 | -1.16 |

Tally:
- 6 of 7 strategies were positive on TRAIN; 1 of 7 was positive on TEST.
- All 14 portfolios lost on TEST. The recommended portfolio is **none**.
- Pooling all 60 days, even with the TRAIN-optimised half included, no strategy reaches t = 2. The full-period t-stats are: gap 1.52, rs_neutral 1.51, index_trend 1.42, low_freq_breakout 1.08, regime_momentum 0.17, orb -0.04, vwap_reversion -0.81.

## 3. What is real and what is likely luck or overfitting

**How big the search was:**
- 106 harness configs on TRAIN (13+11+16+14+19+12+21).
- At least 7 TRAIN event studies that shaped the rules.
- Choices of side made after seeing results: low_freq_breakout went long-only after its shorts lost (-19.0k); gap went short-only.
- 14 portfolio candidates. RG_fit was itself mined from 6 market-state buckets × 7 strategies.
- So the effective number of trials is well over 100.

**Every TRAIN result fits pure noise:**
- With zero true edge, the best t-stat out of 106 independent tries is about 2.53 on average (95th percentile 3.29).
- Within one family of 11–21 configs, the expected best t from noise is 1.6–1.9.
- The best TRAIN t per family was 0.44, 1.23, 0.08, 2.66, 2.28, 1.36 and 2.78 (mean 1.55). Searching noise would produce that.
- Nothing clears a multiple-testing bar of about t 3.3–3.5.

**TRAIN and TEST ranked the strategies in opposite order:**
- The rank correlation between TRAIN t and TEST t across the 7 families is **-0.43** (Pearson -0.38).
- The three strategies with TRAIN t ≥ 2 (gap, rs_neutral, low_freq_breakout) were 3 of the 4 worst on TEST.
- low_freq_breakout went from t +2.78 to -2.28, and its win rate fell from 70% to 33%.
- The harder the search succeeded in-sample, the harder it failed out of sample. That is what overfitting looks like.

**Portfolio t-stats are selection stacked on selection:**
- Simply averaging the 7 stand-alone books gives TRAIN t **3.49**, because averaging weakly correlated in-sample winners (mean pairwise correlation +0.08) raises t mechanically.
- The same average has TEST t **-1.81**.
- TOP_T_shared's TRAIN t of 3.71 means nothing for the same reason. Diversifying cannot create an edge that the parts do not have.

**What is real (mechanical, not fitted):**
- **Cutting turnover cuts losses.** Charges are certain and the signal adds nothing, so fewer trades means a smaller loss.
  - B1 (3 trades a day) cut the loss by 85% on TRAIN and 64% on TEST.
  - It did not change the sign: TEST gross per trade was actually worse (-Rs 323 vs -Rs 161).
- **Cutting gross long exposure cuts beta.** The baseline carries about Rs 15L of effective NIFTY exposure, and that part of the loss scales directly with exposure.
- **The 2.5% target is decorative.** Only 3% of trades hit it.

**What is luck, or not established:**
- Every TRAIN profit figure.
- The "in-play" filter in orb. Moving its threshold 20% (2.0 to 1.6) wipes out the TRAIN profit.
- The gap fade. All 8 ±20% variants were positive on TRAIN and negative on TEST, including after hedging against NIFTY.
- Residual momentum (rs_neutral). TEST gross was only Rs 2.1k against Rs 10.8k of charges.
- The volume-confirmed breakout (low_freq_breakout).
- All four portfolio switching methods.
- **The NIFTY regime gate on the baseline (B2).** It made TRAIN worse per trade (PF 0.73 to 0.46) but nearly removed the TEST loss (-82k to -3k). The effect flips sign between periods, so it is not a rule anyone can rely on.

**index_trend, the only strategy positive on TEST (+13.7k, PF 1.25, t 0.58), is unproven:**
- Resampling TEST days, the chance its true mean is ≤ 0 is 28%.
- Removing its 2 best TEST days turns TEST to -5.9k. Removing its best TRAIN day halves TRAIN, from 37.7k to 19.1k.
- On TEST, shorts made +25.9k and longs lost -12.2k, while NIFTY fell (10:00→15:15 moves summed to -2.82%). TEST daily P&L correlates -0.38 with NIFTY.
- The source of profit changed between periods:
  - TRAIN: stock picking drove it. The strategy made 487bp against 7.8bp for simply trading NIFTY in the signal direction.
  - TEST: the strategy made 226bp, *worse* than the 262bp from trading NIFTY in the signal direction.
- Picking it now means choosing on TEST (1 of 7 families). It deserves a frozen forward test, not capital.

**A 23-day TEST window can reject big edges but cannot confirm small ones:**
- low_freq_breakout's t of -2.28 is strong evidence against its TRAIN edge.
- But a genuine PF-1.2 strategy (daily Sharpe about 0.12) has an expected t over 23 days of only 0.12 × √23 ≈ 0.58. It would show a loss about 28% of the time.

**How much out-of-sample evidence a PF ~1.2 strategy needs:**
- In this study, PF 1.25–1.30 corresponded to a daily Sharpe of 0.12–0.20 (index_trend TEST 0.12, orb TRAIN 0.20).
- The table below is for a single strategy whose rules were frozen in advance.

| Daily Sharpe | Days for t = 2 | Days for 80% power at 5% one-sided | Days if corrected for ~120 trials (t ≈ 3.3) |
|---|---|---|---|
| 0.12 | 278 | 430 | about 750 |
| 0.15 | 178 | 275 | about 480 |
| 0.20 | 100 | 155 | about 270 |

- The answer: **at least about 250 trading days (about one year) of untouched out-of-sample data**, across more than one market regime, before trusting a PF ~1.2 intraday strategy.
- Counted by trades: PF 1.2 at a 50% win rate is a per-trade Sharpe of about 0.09, so about 500 independent trades. But trades on the same day share the market move, so count days, not trades.
- A 4-week paper run is about 8% of the evidence needed.

## 4. Recommendation for the live app

**Change now.** These changes reduce risk, do not depend on any fitted edge, and are all enforced in code, the same way as the daily-loss breaker:

1. **Keep `TRADING_MODE=paper`. Do not go live, and do not pitch this to an investor.**
   - The current rules lose about 0.31% of capital a day: -18.6% in 60 days, -8.2% in the 23 TEST days.
2. **Cap turnover.**
   - Add `MAX_TRADES_PER_DAY=3`, enforced in `place_trade`.
   - Set `MAX_OPEN_POSITIONS=3`. It is 10 in `.env` today, even though the config default is 3.
   - Evidence: B1. Expected charges drop from about Rs 1,525 a day to about Rs 350–475 a day.
3. **Cap gross exposure at 1x capital (Rs 10L)**, even though 5x margin allows Rs 50L. This directly caps the beta (about Rs 15.3k per 1% NIFTY move today).
4. **Allow at most 1 new entry per scan cycle**, to stop the 10:00 burst. This is a risk control, not something tested for P&L.
5. **Instrument everything.** Log for every trade:
   - NIFTY's return over the holding window
   - the fill price vs the reference price, i.e. measured slippage
   - charges, checked against the cost model
   - a daily alpha figure: P&L minus beta × NIFTY

**Do not:**
- Add the NIFTY regime gate as a "fix". B2 contradicts itself between TRAIN and TEST.
- Add shorts to the live app. Short-side profits here mostly came from short beta in a falling market; NIFTY fell in both windows, and the gap fade's shorts lost on TEST even after hedging. Shorts also bring practical risks: MIS auto-square-off penalties, circuit-lock risk and broker short restrictions.
- Re-tune the stop, target or exit time. Keep the 15:15 square-off, because it prevents overnight carry.
- Deploy any of the 7 strategies or 14 portfolios.
- Use 1/7 capital slices. Orders of about Rs 28.6k push brokerage to about 0.07% per side, versus about 0.01% on Rs 2L orders.
- Use shared-account stacking. It stacks every strategy's 0.5% risk at full size, which is how TEST drawdowns reached 5–10.6%.
- Re-fit on TRAIN and TEST combined.
- Assume the LLM debate layer adds an edge. It was never backtested.
- Read a good 4 weeks as proof.

**4-week paper validation plan** (20 regular sessions; exclude any special or Muhurat session). Four weeks can only catch failures, not prove an edge. A strategy with zero edge and 3 trades a day would end those 20 days at about -Rs 9k ± 19k (1 SD, based on B1's daily SD of about Rs 4.2k).

- **Track A: live app in paper mode** with changes 2–4, parameters frozen for the 4 weeks.
- **Track B: index_trend shadow.** Use the exact current `strategies/index_trend.py`, frozen and recorded by git hash. Run the harness each evening on that day's freshly downloaded bars. Expect about 10–13 signal days.
- **Track C: the unchanged baseline**, replayed in the harness as a control.

Pass/fail criteria. All P1–P5 must pass:

| Check | Pass | Fail |
|---|---|---|
| P1 Replay fidelity (Track A) | ≥90% of paper trades match the harness replay of the same day (same symbol and side, entry within ±1 bar) | Below 90%, or mismatches that cannot be explained |
| P2 Slippage | Median ≤ 2bp, 90th percentile ≤ 5bp vs the next bar's open | Median > 3bp. Then re-run every result at the measured slippage before anything else |
| P3 Charges | Each trade within ±10% of the cost model (spot-check 5 trades against Groww's calculator) | Any systematic gap above 10% |
| P4 Risk limits | Zero breaches: ≤3 entries/day, ≤Rs 2L per name, ≤Rs 10L gross, flat by 15:20, no overnight positions | Any breach |
| P5 Turnover | Average ≤3 trades/day, charges ≤Rs 500/day | Above either |
| A kill switch | 4-week net above -4% (-Rs 40k), and no day worse than -2.2% | Stop and debug if either is breached. -4% is about 1.6 SD below what a no-edge book would do, so false alarms are about 5% |
| B: keep forward-testing (not a go-live signal) | Net ≥ 0 after charges and 2bp slippage; PF ≥ 1.0; ≥8 signal days including both long and short days; the stock basket beats the trade-NIFTY-in-the-signal-direction comparison on ≥50% of signal days | Any one fails: archive it, or re-specify it as an index trade |
| C control | Keeps losing about 0.3%/day | If it turns profitable, the regime has changed; re-check all conclusions |

**Go-live gate (not at week 4):**
- At least 250 trading days of untouched data (forward paper trading, plus newly acquired history never used in design).
- t ≥ 2.0 and PF ≥ 1.2 at 2bp slippage.
- Positive in both a NIFTY up-trend month and a down-trend month.
- No single month contributing more than 40% of profit.
- Then start at 10–20% of capital.

**Highest-value research step:** get at least 2 years of 5-minute NSE data. yfinance caps 5-minute history at 60 days, and that cap, not ideas, is what limits this study. Then re-run the 7 frozen designs walk-forward.

## 5. Missing or still unverified

- **One regime, few days.** 60 sessions, with NIFTY -2.0% over TRAIN and -1.1% over TEST (close to close). No up-trending month was tested. Every short-tilted result is suspect for this reason.
- **TEST is 23 days, not 24.** 2026-09-14 is missing from the cache; it is a data gap, not a holiday. The end-of-day bars are also irregular: 43 days end at 15:15, 12 at 15:25, and 5 at 15:10, where positions exit at the last bar's close.
- **Data quality.**
  - The universe is today's NIFTY 100, so there is survivorship bias.
  - The 09:15 bars have about zero volume, which affects the opening-range volume filters.
  - "Previous close" is the last 5-minute close, not the official close.
  - Corporate actions were not checked.
- **Slippage of 1bp/side is assumed, not measured.** The live app also decides on a 15-minute LLM cycle, so filling at the next 5-minute bar's open is optimistic for it.
- **The baseline is only an approximation of the live app.**
  - It has no LLM, scanner or debate layer.
  - It hard-blocks signals without volume confirmation; the live `save_signal` only warns.
  - It evaluates every 5-minute bar; the live app scans every 15 minutes.
- **The cost model has not been checked against a real Groww contract note** (brokerage, minimums, current exchange rates).
- **Max DD is measured on end-of-day equity.** Intraday drawdown is larger, and a gap can push a day past the -2% limit.
- **Short-selling frictions are not modelled:** MIS square-off timing and penalties, circuit locks, and restricted or banned names.
- **The t-stats assume independent days.** Profits are fat-tailed: the top 5 trades are 33–85% of TRAIN net in several strategies, so the real sample is smaller than it looks. The designers computed no bootstrap confidence intervals; I computed one, for index_trend only.
- **Not independently verified by me:** the portfolio Multi wrapper's rule for who gets priority under the shared 10-position cap (I relied on its own self-check), and the designers' event-study scripts.
- **Reproducibility.** The config logs and event-study scripts live in a temporary scratchpad, not in the repo. They should be committed.
- **Two simulators that disagree, never reconciled.** The older `alpha_prime/backtest` reported +1.09% net for SL 1.2 / target 2.5 on 12 NIFTY-50 stocks (May–Aug). This harness gives -18.6% for the same exits on NIFTY-100 top gainers (Jul–Oct). The universe, period and selection all differ, but the two have never been run on a common window.

Files:
- B1–B3 variant script: `C:\Users\SREENI~1\AppData\Local\Temp\claude\C--Users-Sreenivasan-OneDrive-Desktop-Algo-Trading\08ef454e-05e3-4e27-b861-1677f760d63c\scratchpad\review\baseline_variants.py`
- B1–B3 results: `...\scratchpad\review\baseline_variants.json`
- Daily P&L for all 7 strategies plus the baseline, with NIFTY returns: `...\scratchpad\review\daily_all.csv`
- Results the study produced: `E:\Algo Trading\research\strat_lab\results\`