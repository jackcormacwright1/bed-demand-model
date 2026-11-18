# Methodology

This document sets out how the bed demand model works: the synthetic hospital it is tested on, the original method it rebuilds, each change made to that method and why, how the simulation runs, and how the results were checked. Every symbol is defined where it first appears, and the main calculations have small worked examples.

**All data is synthetic.** Admission rates, lengths of stay, long-stay shares and the rest are invented, with values chosen to be plausible rather than taken from any organisation. The project rebuilds, from scratch, a bed model first written for NHS use; no NHS code, data or results are included. Because the process that generates the synthetic data is known, every estimate can be checked against the true answer, which real data never allows.

Code: [github.com/jackcormacwright1/bed-demand-model](https://github.com/jackcormacwright1/bed-demand-model)

---

## 1. The question

*Given one year of inpatient spells, how many beds does each ward type need next year, so that it has enough on a chosen share of nights?*

Two numbers answer it for each ward type:

- **Core demand**: enough beds for 85% of nights. Plan routine staffing and beds around this.
- **Escalation demand**: enough for 95% of nights. The gap between the two is what a ward occasionally needs to flex up to.

The model also combines ward types that could share a ward, tests ways of reducing demand (shorter stays, same-day emergency care, day cases and others), and projects demand forward several years.

---

## 2. The synthetic hospital

Code: [`synthetic.py`](https://github.com/jackcormacwright1/bed-demand-model/blob/main/src/bedmodel/synthetic.py)

### 2.1 Ward types

Fifteen surgical and related ward types, each **emergency** (non-elective) or **elective**. Each has an average number of admissions a day, $\lambda$ (lambda), and a length-of-stay distribution with two parts (section 2.3). The reference values:

| Ward type | Kind | Admissions a day | Typical stay (days) | Long stays (% of admissions) |
|---|---|---|---|---|
| General surgery, emergency, male / female | emergency | 4.0 / 3.8 | 3.0 / 2.8 | 2.5 |
| General surgery, elective, male / female | elective | 1.6 / 1.5 | 1.8 | 1.0 |
| Trauma and orthopaedics | emergency | 5.0 | 4.5 | 6.0 |
| Gynaecology, emergency / elective | emergency / elective | 1.8 / 1.2 | 1.5 | 0.5 |
| Urology, emergency male / female, elective | emergency / elective | 1.6 / 0.6 / 1.4 | 2.5 / 2.2 / 1.5 | 2 / 2 / 0.5 |
| Gastroenterology, male / female | emergency | 2.2 / 2.0 | 4.0 | 4.0 |
| Colorectal surgery, male / female | emergency | 0.9 / 0.8 | 6.0 | 4.0 |
| Breast surgery | elective | 0.5 | 1.3 | 0 |

"Typical stay" is the median of the main group of patients before rounding up to whole days. Each **synthetic hospital** (numbered 0, 1, 2, …) varies these at random: admissions by ±15%, typical stays by ±10%, the long-stay share by a factor of 0.6 to 1.4, and the long-stay median by ±15%.

### 2.2 Admissions

The number of admissions to ward type $s$ on day $t$ is **Poisson** (a count of independent events with a fixed average) with mean

$$\lambda_s \times w_s(t) \times f_{\text{kind}}(t) \times m(t)$$

- $w_s(t)$, the **day-of-week weight**: elective admissions happen on weekdays only (weights 1.4 Monday to Friday, 0 at weekends); emergency admissions are busiest on Mondays (1.12) and quietest on Saturdays (0.88). Weights average 1 over a week.
- $f(t)$, the **season**: emergency admissions follow $1 + 0.10\cos\big(2\pi(d - 15)/365.25\big)$, with $d$ the day of the year, so they peak 10% above average in mid-January and dip 10% below in mid-July. Elective admissions fall to 75% in August and 40% over Christmas (22 December to 2 January).
- $m(t)$, the **surge**: a pressure factor shared by every emergency ward type. It wanders slowly: $m(t) = e^{x_t - \sigma^2/2}$, where $x_t = \rho\, x_{t-1} + \varepsilon_t$ is an **AR(1) process** ("autoregressive of order 1": each day's value is $\rho$ times yesterday's plus fresh noise $\varepsilon_t$). Here $\rho = 0.92$, so a busy spell typically lasts one to two weeks, and $x_t$ has standard deviation $\sigma = 0.10$, so the multiplier is usually within ±10% of 1. The $-\sigma^2/2$ makes it average exactly 1. Elective admissions are scheduled and don't surge.

### 2.3 Length of stay

A patient's stay $L$ is in whole days, at least 1, and means the patient occupies a bed on $L$ midnights. It comes from one of two groups:

- **most patients**: a **lognormal** stay (its logarithm is normally distributed, which gives a right-skewed shape) with the ward type's typical stay as its median and a log-scale standard deviation of 0.5 to 0.85;
- **long stays**, a few percent of admissions (complex discharges, waits for care packages or placements): lognormal with a median of 30 to 50 days and log-scale standard deviation 0.45. About a quarter of them stay more than 60 days.

Continuous stays are rounded up to whole days. Long stays are rare but heavy: in hospital 0, the 0.9% of spells over 60 days account for 12% of all bed-days, and in trauma and orthopaedics, 58 of 2,193 finished spells (2.6%) account for 23% of the ward type's bed-days.

### 2.4 Flags for the scenarios

Each spell carries flags used by the demand-reduction scenarios (section 7.3), generated plausibly: days spent in hospital after being ready to leave (30–70% of a long stay; 1–3 days for 6% of other stays); readmission (8% of emergency admissions); eligible for same-day emergency care (SDEC: a share of emergency stays of one or two days); planned as a day case but stayed overnight (30% of one-day elective stays); eligible for a day case (40% of elective stays of one or two days); and eligible for a virtual ward (a share of emergency stays of three days or more).

### 2.5 The extract

One year of admissions (1 April 2025 to 31 March 2026) is simulated, and an extract is taken 30 days after the year ends, as a spell query would return it: every spell admitted in the year, with its stay if it had finished by the extract date, or its **stay so far** (and "still in hospital") if it hadn't. In hospital 0, 20 of 11,294 spells are still in hospital, 11 of them in trauma and orthopaedics. The extract's true stays are kept separately, for checking only.

---

## 3. The original method

The model being rebuilt (in summary; it is not reproduced here):

1. **Extract**: finished spells of 1 to 60 days, by ward type, from one financial year. Spells of more than 60 days are excluded in the query's WHERE clause, so they are missing from the admission counts as well as the stays.
2. **Arrivals**: $\lambda_s$ = spells ÷ days in the year; day-of-week weights from the share of admissions on each weekday. Poisson admissions each day; no seasonality or surges.
3. **Length of stay**: stays above **Tukey's far fence** (section 4.2) are excluded from the fit. Five distributions (lognormal, exponential, gamma, Weibull, log-logistic), each anchored at 0 and at 1 day, are fitted by maximum likelihood, and the one with the smallest **Kolmogorov–Smirnov (KS) statistic** (the largest gap between the fitted and observed cumulative distributions) is kept. If even that gap is 0.10 or more, the model resamples the observed stays instead. Sampled stays are rounded and capped at 90 days.
4. **Simulation**: 5,000 simulated years per ward type, each starting from an empty hospital 30 days before the year.
5. **Outputs**: the 85th and 95th percentiles of occupancy, interpolated between whole numbers; for ward types needing fewer than 5 beds, a "continuous curve percentile" that interpolates each day's occupancy to hourly steps; bootstrap confidence intervals over the simulated years; ward groups; scenarios; multi-year projection; discharge targets.

The engine (Poisson admissions, sampled stays, occupancy from running totals, percentiles over all simulated nights, ward groups from summed occupancy) is sound and is kept. The changes are in what the engine is given.

---

## 4. What changed, and why

Code: [`los.py`](https://github.com/jackcormacwright1/bed-demand-model/blob/main/src/bedmodel/los.py), [`arrivals.py`](https://github.com/jackcormacwright1/bed-demand-model/blob/main/src/bedmodel/arrivals.py), [`model.py`](https://github.com/jackcormacwright1/bed-demand-model/blob/main/src/bedmodel/model.py)

Each change can be switched on or off by itself in the dashboard (*Method → Custom*), and section 8 measures what each is worth against the truth.

### 4.1 Keep stays over 60 days

A stay of more than 60 days occupies a bed for more than 60 nights. Dropping those spells from the extract removes both their admissions and their bed-days, and the fitted distribution's tail can't stand in for them, because it is fitted to data that doesn't contain them.

> **Worked example.** In hospital 0's trauma and orthopaedics, 58 finished spells lasted more than 60 days. They are 2.6% of the ward type's spells but hold 23% of its bed-days. The ward type averages about 62 occupied beds, so those patients fill about 14 of them on an average night, and the original extract never sees them.

**Is a cutoff ever justified?** Only if those patients are planned for somewhere else, for example on a dedicated long-stay or complex-discharge ward that is sized separately. Then the cutoff should be applied to the long-stay patients' admissions and stays together (as the query does), and the long-stay ward sized from the same spells. For the beds a ward type needs in total, no.

### 4.2 Don't exclude Tukey outliers

**Tukey's far fence** is $Q_3 + 3 \times \text{IQR}$, where $Q_1$ and $Q_3$ are the 25th and 75th percentiles of the stays and $\text{IQR} = Q_3 - Q_1$ is the interquartile range. The original excluded stays above it from the fit, while still counting their admissions, on the grounds that very long stays are often outlier patients on another specialty's ward.

> **Worked example.** Hospital 0, general surgery, emergency, male: $Q_1$ = 2 days, $Q_3$ = 6 days, so $\text{IQR}$ = 4 and the fence is $6 + 3 \times 4 = 18$ days. Stays over 18 days are 3.5% of spells but 24% of bed-days. In trauma and orthopaedics the fence is 27 days, and stays beyond it are 6.8% of spells holding 41% of bed-days.

For a right-skewed distribution like length of stay, the far fence isn't a filter for errors: it cuts into the real tail, which is where much of the bed demand is. The excluded patients' admissions still count, but each is given a typical stay instead of a long one. If some long stays really are boarders from elsewhere, the fix is to attribute them to the ward type they belong to, not to shorten them.

### 4.3 No 90-day cap

Sampled stays were capped at 90 days. With long stays kept, a few genuinely last longer (up to 200 days in hospital 0's data). The cap matters little once the tail is otherwise intact (section 8), but there is no reason to keep it.

### 4.4 A year-long warm-up

Each simulated year starts from an empty hospital some days before the year being measured, and those warm-up days are discarded. On the first measured night, the only patients in hospital are those admitted during the warm-up, so a warm-up shorter than the long stays starts the year with too few of them.

> **Worked example.** One admission a day, every stay 90 days. In the long run 90 beds are occupied (section 5.2). After a 30-day warm-up, the first measured night has only the 31 patients admitted in the last 31 days. Occupancy climbs to 90 only on day 60 of the year. With a 365-day warm-up, it is 90 from the first night.

The recommended method uses a 365-day warm-up that repeats the planning year's own pattern.

### 4.5 Use spells still in hospital (Kaplan–Meier)

An extract of finished spells taken soon after the year ends misses patients admitted near the end of the year who are still in hospital, and those are disproportionately the long stays. Dropping them shortens the average stay.

The fix is the **Kaplan–Meier estimator**, the standard tool from survival analysis for data in which some times are only known to be "at least this long" (**censored**). One way of computing it, Efron's "redistribute to the right", reads naturally as weights on the spells:

1. Sort the spells by stay. Each starts with weight $1/n$.
2. Work from the shortest stay to the longest. When a spell still in hospital is reached, all we know is that its stay is longer than this, so share its weight equally among all the spells with longer stays.
3. The finished spells end up with all the weight, and new patients' stays are drawn from them with those probabilities.

At a tie, finished spells count as shorter than unfinished ones.

> **Worked example.** Four spells: stays of 1 and 2 days (finished), 2 days so far (still in hospital), and 3 days (finished). Each starts with weight 0.25. The unfinished spell's 0.25 passes to the only longer spell, the 3-day stay. Weights: 0.25, 0.25, 0, 0.5. The average stay is $0.25 \times 1 + 0.25 \times 2 + 0.5 \times 3 = 2.25$ days, against 2.0 from the finished spells alone. This is the same as the usual Kaplan–Meier calculation: the share still in hospital after 1 day is 3/4, after 2 days $3/4 \times 2/3 = 1/2$ (one of the three at risk finishes), and after 3 days 0.

The same weights carry each spell's flags, so the scenarios still act on individual patients. In a test with 20,000 stays and follow-up times between 5 and 60 days, the finished spells alone underestimated the average stay by more than 10%, and the Kaplan–Meier weights recovered it to within 2%.

### 4.6 Resample spells rather than fit a distribution

The recommended method draws new patients' stays by resampling the (weighted) spells of the data year rather than fitting a distribution. Reasons:

- The stays are whole days with many ties, so the Kolmogorov–Smirnov statistic against a continuous distribution is inflated by the ties themselves, and choosing among five distributions by it mostly rewards fitting the short stays well.
- A distribution fitted to the bulk of the stays need not get the tail right, and the tail holds a large share of the bed-days (section 4.1).
- Resampling keeps each patient's flags, so the scenarios can act on individual patients (section 7.3).

With one year of data the resampled distribution is itself uncertain, especially in the tail; section 6.2 carries that uncertainty into the bed numbers.

**Shortening stays.** A scenario that shortens stays by a share $r$ can't simply multiply whole days and round: a 10% cut takes a 4-day stay to 3.6 days, which rounds back to 4, so no stay of 4 days or less changes. Instead, a stay recorded as $L$ whole days is treated as continuous, somewhere in $(L-1, L]$, uniformly: $L - U$, with $U$ uniform on $(0, 1)$. It is multiplied by a factor $k$ and rounded up again. A stay can't fall below one night, so to cut the average whole-day stay by exactly $r$, the longer stays have to shorten a little more than $r$: $k$ is found by bisection (repeatedly halving the interval that must contain it) so that the average of $\lceil k(L - U) \rceil$, at least 1, equals $(1 - r)$ times the current average.

> **Worked example.** A 5% cut ($k \approx 0.95$) and an 8-day stay. The shortened stay is at most 7 days when $(8 - U) \times 0.95 \le 7$, that is when $U \ge 8 - 7/0.95 = 0.632$: a 37% chance. Rounding $8 \times 0.95 = 7.6$ instead gives 8 every time.

### 4.7 Seasonality and surges

The original's only pattern in admissions was the day of the week. Real demand also has a season and busy spells that last days or weeks, and both widen the range of occupancy and so raise the 85th and 95th percentiles. One ward type's year is too noisy to estimate them, so they are estimated from the daily totals of all ward types of each kind, and shared.

**Seasonality.** For emergency admissions, one smooth yearly wave, $1 + a\cos(2\pi d/365.25) + b\sin(2\pi d/365.25)$, with $a$ and $b$ found by least squares (the values that minimise the sum of squared gaps) from the ratio of observed to expected admissions each day. Its size is $\sqrt{a^2 + b^2}$. For elective admissions, one factor per month: admissions in the month ÷ admissions expected from the weekly pattern alone. In hospital 0 the emergency wave is 8.9% (true: 10%), peaking on 4 January (true: 15 January); the elective factors are 0.80 for August and 0.78 for December.

**Surges** are estimated by the **method of moments** (choosing the parameters so that the model's averages match the data's). With $e_t$ the expected emergency admissions on day $t$ (from the rates, weekly pattern and season) and $o_t$ those observed, let $d_t = (o_t - e_t)/e_t$, the relative excess. Chance alone (Poisson) gives $d_t$ a variance of $1/e_t$, and a surge multiplier with variance $s^2$ adds $s^2$. So

$$s^2 = \text{mean}\big(d_t^2 - 1/e_t\big).$$

> **Worked example.** If about 25 emergency admissions are expected a day, chance alone gives $d_t$ a variance of $1/25 = 0.04$ (a standard deviation of 20%). If the observed $d_t^2$ average 0.047, the surge accounts for $s^2 = 0.047 - 0.040 = 0.007$, a standard deviation of about 8%. That is why surges are hard to see day by day: chance is the bigger part.

How long surges last is estimated from whole weeks, which is steadier than comparing one day with the next. The same calculation on weekly totals gives the variance of a week's average surge, which is $s^2 \times B(\rho)$, where

$$B(\rho) = \frac{1}{49}\Big(7 + 2\sum_{\ell=1}^{6} (7 - \ell)\,\rho^{\ell}\Big)$$

is the share of the daily variance that survives averaging over 7 days: $1/7 = 0.14$ if days are independent ($\rho = 0$), 0.80 if $\rho = 0.9$, and 1 if surges last for weeks. Solving for $\rho$ gives the persistence. The surge's log-scale standard deviation is then $\sigma = \sqrt{\log(1 + s^2)}$, which is close to $s$ when $s$ is small. In hospital 0 the estimates are $\sigma = 0.083$ and $\rho = 0.65$ (true: 0.10 and 0.92). One year holds only a dozen or so independent surges, so these are rough, and across 20 hospitals they vary a lot; the data bootstrap (section 6.2) carries that into the bed numbers. Surges matter less than the long stays: they add about 4 core and 7 escalation beds to hospital 0's true requirement of 209 and 224.

### 4.8 Whole beds

Core demand is reported as the smallest whole number of beds $b$ with occupancy at or below $b$ on at least 85% of simulated nights, together with the share of nights $b$ actually covers. The original interpolated between whole numbers, and for ward types needing fewer than 5 beds interpolated each night's occupancy to hourly steps (the "continuous curve percentile"), which gives fractional beds.

> **Worked example.** Eight simulated nights with occupancy 1, 1, 2, 2, 3, 3, 4, 4. The interpolated 85th percentile is 3.95 beds. In whole beds: 3 beds cover 6 of 8 nights (75%), so 85% needs 4 beds, which cover all 8 (100%). "3.95" suggests a precision that isn't there; "4 beds, covering 100% of nights" says what opening 4 beds buys. For a small ward type, the useful comparison is the share of nights each whole number of beds covers.

---

## 5. The simulation engine

Code: [`simulate.py`](https://github.com/jackcormacwright1/bed-demand-model/blob/main/src/bedmodel/simulate.py)

### 5.1 Occupancy from running totals

For each simulated year and each day $t$ (warm-up included), the number of admissions is Poisson with mean $\lambda_s\, w_s(t)\, f(t)\, m(t)$, and each admission gets a stay $L$ from the ward type's model. A patient admitted on day $a$ with stay $L$ is in a bed on the midnights of days $a, a+1, \dots, a+L-1$ and is discharged on day $a + L$. So midnight occupancy is

$$O_t = (\text{admissions on days} \le t) - (\text{discharges on days} \le t).$$

> **Worked example.** A patient admitted on Monday with a 3-day stay is counted on Monday, Tuesday and Wednesday nights and discharged on Thursday.

Both running totals are computed for every simulated year at once (with `numpy.bincount` and a cumulative sum), so there is no loop over patients. Ward types are simulated together and share the surge paths, so their occupancies rise and fall together as they would. Each ward type has its own random number stream, so a ward type's results don't depend on which others are simulated, and the same seed always gives the same answer.

### 5.2 A check: Little's law

Once a simulation has warmed up, average occupancy should equal admissions a day × average stay (**Little's law**).

> **Worked example.** Hospital 0, general surgery, emergency, male: 4.14 admissions a day × 5.55 days = 23.0 beds. The simulation's average is 23.05.

The tests check this exactly for fixed stays, and that with Poisson admissions and fixed stays the occupancy on a night has variance equal to its mean, as theory says it should (it is Poisson too).

---

## 6. Uncertainty

### 6.1 Monte Carlo noise

The original reported a bootstrap confidence interval over its simulated years: how far the percentile would move if the simulation were rerun with new random numbers. With thousands of simulated years this is small, typically a bed or less, and running more simulations shrinks it further. It says nothing about how well one year of data describes the hospital.

### 6.2 Uncertainty from having one year of data

The recommended method also refits the model to bootstrap copies of the data year and reruns it (`data_interval`). A **bootstrap** copy is a new dataset drawn at random from the old one; here it is a **local block bootstrap**: the year is cut into weeks, and each week is replaced by a week drawn at random from within 4 weeks of it, moved whole (its admissions, their stays and flags). Drawing nearby weeks keeps the season in place; moving whole weeks keeps busy spells together and keeps every admission on its weekday.

Redrawing weeks can't show one thing: that the whole year may have been busier or quieter than usual. A surge lasts longer than a week, so the year's average surge level is itself uncertain, and every bootstrap copy inherits the data year's. So each copy's emergency admission rates are also multiplied by $e^{zs - s^2/2}$, with $z$ a standard normal draw and $s$ the standard deviation of the average of 365 consecutive days of the fitted surge process:

$$s = \sigma\sqrt{\frac{1}{T}\Big(1 + 2\sum_{\ell=1}^{T-1}\big(1 - \tfrac{\ell}{T}\big)\rho^{\ell}\Big)}, \qquad T = 365.$$

> **Worked example.** With the true surge settings, $\sigma = 0.10$ and $\rho = 0.92$, this gives $s = 0.025$: a year's emergency admissions can easily run 3% above or below the long-run rate because of surges alone. Across the 20 synthetic hospitals the data years' average surge levels ranged from 0.96 to 1.08. In hospital 0 the fitted settings ($\sigma = 0.083$, $\rho = 0.65$) give only $s = 0.009$, because one year's estimate of the persistence is rough.

The range of bed numbers across 30 copies (the middle 95%) is the uncertainty that comes from the data.

> **Worked example.** Hospital 0, core beds for all ward types: 213 from the data year, with a bootstrap range of 205 to 222. The true number is 209. The original-style Monte Carlo interval is about one bed wide and doesn't contain it.

Section 8.2 checks across 20 hospitals whether these ranges contain the truth as often as they claim.

---

## 7. Outputs

### 7.1 Core and escalation beds

As section 4.8, for each ward type and for all ward types together. The total comes from adding the simulated occupancies night by night, not from adding the ward types' bed numbers.

### 7.2 Ward groups

Ward types that could share a ward. A shared ward needs fewer beds than its ward types separately, because they aren't all busy on the same nights (the **diversity** effect). The combined requirement comes from the summed occupancy, night by night.

> **Worked example.** Hospital 0: colorectal surgery (male 13, female 12 core beds) and gastroenterology (male 20, female 21) need 66 beds as four separate wards and 59 as one shared ward: 7 beds saved.

The same effect is why the total for all ward types comes from summed occupancy: in hospital 0, the fifteen ward types' core numbers add up to 244 beds, but the hospital as a whole needs 213 on 85% of nights.

Because emergency ward types share surges, they are busy together a little more often than if they were simulated independently, as the original did, so the saving from sharing is slightly smaller: by about one bed across hospital 0's four groups.

### 7.3 Scenarios

The original model's scenarios, plus three it had switched off. With the recommended method, each acts on individual simulated patients:

| Scenario | Effect on a simulated patient |
|---|---|
| Stays 5% or 10% shorter | Stay shortened as in section 4.6 |
| SDEC takes half / all its eligible patients | An SDEC-eligible patient doesn't take a bed (with probability ½ or 1) |
| Planned day cases go home the same day | A patient planned as a day case doesn't take a bed |
| All day-case-eligible patients go home the same day | A day-case-eligible patient doesn't take a bed |
| Readmissions down 20% | A readmission doesn't happen, with probability 0.2 |
| No waiting once ready to leave | Stay shortened by the days spent waiting after being ready to leave (at least 1 night) |
| Virtual ward saves 3 days | An eligible patient's stay is 3 days shorter (at least 1 night) |
| Stays of 60+ days cut to 45 | A stay of 60 days or more becomes 45 |

With the original fitted method, the scenarios work as the original did: removing a share of patients becomes a cut in the admission rate (the flagged share × the share removed), and shortening some stays becomes an equivalent cut in the average stay, applied to every patient.

> **Worked example.** In hospital 0, 5.2% of admissions are SDEC-eligible. Cutting every ward type's admissions by its SDEC-eligible share removes 5% of patients of every kind, long stays included, and saves 8 core beds (213 to 205). Removing the SDEC-eligible patients themselves, who stay one or two nights, saves 3 (213 to 210).

### 7.4 Multi-year projection

For year $t$ after the planning year: admissions × $(1 + g)^t$, with $g$ the yearly growth (1.5% by default), and every stay × $(1 - i)^t$, with $i$ the yearly improvement in length of stay (1% by default), applied as in section 4.6. The mix of ward types, the weekly pattern and the season stay as in the data year.

### 7.5 Discharge targets

As in the original, an operational translation of a length-of-stay cut: the extra discharges a week, after stays of 7 days or less, implied by shortening each finished spell by the cut. Each whole-day stay is treated as continuous within its day (section 4.6), so a stay of $L$ days becomes short with probability $1 - \min\big(\max(L - 7/(1 - r), 0), 1\big)$. The original rounded instead, which for a 5% cut adds no short stays at all.

---

## 8. Results against the truth

Code: [`scripts/run_methods.py`](https://github.com/jackcormacwright1/bed-demand-model/blob/main/scripts/run_methods.py), [`scripts/run_intervals.py`](https://github.com/jackcormacwright1/bed-demand-model/blob/main/scripts/run_intervals.py)

### 8.1 Each change, one at a time

For each of 20 synthetic hospitals, one year of data is observed, each method is fitted to it and simulated (1,000 years), and the result is compared with the **truth**: the following year simulated 1,000 times from the process that generated the data. "Nights covered" is the share of the true year's nights on which the recommended number of beds is enough; the core number aims for 85%. The steps are cumulative, so each row adds one change to those above it, and a change's effect depends on the order. Means across hospitals, ± two standard errors.

| Method | Core beds vs truth (%) | Nights the core number covers (aim 85%) | Escalation beds vs truth (%) | Nights the escalation number covers (aim 95%) | Ward types: average size of error in core beds (%) |
|---|---|---|---|---|---|
| Original method | -21.3 ± 1.7 | 20 ± 5 | -22.6 ± 1.6 | 32 ± 6 | 11.8 ± 1.0 |
| + keep stays over 60 days | -17.9 ± 1.8 | 30 ± 6 | -19.3 ± 1.7 | 44 ± 6 | 10.2 ± 1.1 |
| + no Tukey exclusion | -10.9 ± 1.5 | 53 ± 5 | -12.6 ± 1.4 | 68 ± 5 | 6.7 ± 0.9 |
| + no 90-day cap | -10.2 ± 1.5 | 56 ± 5 | -12.0 ± 1.4 | 70 ± 4 | 6.5 ± 0.9 |
| + 365-day warm-up | -9.6 ± 1.5 | 58 ± 5 | -11.4 ± 1.5 | 72 ± 4 | 6.2 ± 0.9 |
| + resample spells instead of fitting | -5.3 ± 1.5 | 71 ± 4 | -7.3 ± 1.5 | 83 ± 3 | 5.1 ± 0.8 |
| + use spells still in hospital (Kaplan–Meier) | -2.9 ± 1.5 | 78 ± 4 | -5.0 ± 1.5 | 87 ± 3 | 4.7 ± 0.9 |
| + seasonality | -0.1 ± 1.8 | 84 ± 4 | -1.3 ± 1.9 | 93 ± 2 | 4.9 ± 1.0 |
| + surges (recommended method) | +1.0 ± 1.8 | 86 ± 3 | +0.8 ± 2.0 | 95 ± 2 | 5.0 ± 0.9 |
| *On their own:* | | | | | |
| Recommended with the 60-day cutoff | -8.6 ± 1.7 | 61 ± 6 | -8.6 ± 1.9 | 79 ± 5 | 6.8 ± 0.8 |
| Recommended with the Tukey exclusion | -21.4 ± 1.5 | 20 ± 4 | -21.1 ± 1.6 | 37 ± 6 | 15.3 ± 1.0 |

- **The original method understates beds by about a fifth.** For the hospital as a whole, its core number is 21% below the true requirement, and in the true year it is enough on only 20% of nights rather than 85%. The recommended method is within about 1% and covers 86% of nights; its escalation number covers 95% against an aim of 95%.
- **The long-stay tail does most of the damage.** Keeping stays over 60 days recovers 3.4 percentage points of the shortfall; dropping the Tukey exclusion 7.0; resampling spells instead of fitting 4.3; using spells still in hospital 2.4; seasonality 2.8; surges 1.1. The 90-day cap and the short warm-up cost about 0.7 and 0.6.
- **Either trimming choice on its own breaks an otherwise sound model.** Put the 60-day cutoff alone back into the recommended method and it is 9% short, covering 61% of nights. Put the Tukey exclusion alone back and it is 21% short, covering 20%. In the original method the Tukey exclusion's effect was partly hidden: when no distribution fitted well enough (KS of 0.10 or more), the original fell back to resampling every stay, outliers included, and on whole-day stays that happened for 108 of 150 ward-type fits across 10 hospitals. The exclusion bit only where a distribution was accepted.
- **Ward types individually.** With the recommended method, each ward type's core number is off by 5% on average, either way, and covers 88% of true nights on average: slightly cautious, because small ward types round up to whole beds.

These sizes depend on the synthetic hospitals' long-stay shares (0–6% of admissions, invented); the direction of each effect doesn't.

### 8.2 Are the uncertainty ranges honest?

INTERVALS_TABLE

---

## 9. Checking the work

Code: [`tests/`](https://github.com/jackcormacwright1/bed-demand-model/tree/main/tests) (22 automated tests, run with `pytest`)

- **Occupancy.** A 3-day stay occupies exactly three midnights; a removed admission occupies none; a stay running past the end of the year is still counted.
- **Little's law.** With constant admissions and fixed stays, average occupancy equals admissions × stay to within 1%, and the night-to-night variance equals the mean, as for a Poisson count.
- **Warm-up.** With 90-day stays, a 30-day warm-up starts the year at about 31 patients and a 365-day warm-up at about 90.
- **Kaplan–Meier.** The weights match the hand calculation in section 4.5, are equal when nothing is censored, and recover the true average stay where finished spells alone fall more than 10% short.
- **Tukey fence and trimming**, **whole beds and coverage**, and **shortening stays** (the average whole-day stay falls by the share intended, to within half a percentage point; rounding whole days leaves stays of up to 4 days unchanged).
- **The extract.** Unfinished spells show less than their true stay; finished ones show exactly it; long stays exist.
- **The original method** sees only finished spells of 60 days or less; its fit respects the 90-day cap.
- **Scenarios** act on individual patients: SDEC removes exactly the eligible share, and only short stays; removing waiting days shortens stays but never below one night.
- **Ward groups** never need more beds shared than separately.
- **Seasonality and surges** are recovered from five years of data: the emergency wave's size and peak, a persistence above 0.6 and a surge size between 0.05 and 0.15, and the August and Christmas dips in elective admissions.
- **The bootstrap year** keeps every admission on its weekday and within the year, and elective admissions on weekdays.
- **Projection.** Five years of 2% growth raise average occupancy by $1.02^5$, and five years of 2% shorter stays lower it by $0.98^5$, each to within 2%.
- **Data bootstrap.** The range is ordered and covers every ward type, and the main estimate sits inside it (allowing for Monte Carlo noise). The formula for the spread of a year's average surge level matches 3,000 simulated years to within 8%.
- **Discharge targets** match the hand calculation in section 7.5.
- **End to end.** On hospital 0, the recommended method is within 8% of the true core beds, the original method more than 10% below, and keeping the long stays alone moves the original up.

---

## 10. Assumptions and limitations

| Assumption | Why | Likely effect |
|---|---|---|
| All data is synthetic | No real data is used | The results show mechanisms and their sizes in an invented hospital, not figures for any real one |
| The year ahead looks like the data year (apart from the projection's growth and improvement) | One year of data | Trends within the year, changes in case mix and one-off events are not captured |
| Demand, not what happens when beds run out | As in the original | Boarding, cancelled operations and delayed admissions are not modelled; the model says how many beds avoid them |
| Ward types are fixed and each spell belongs to one | As in the original | Patients who move between wards during a spell are counted against the ward type of their spell |
| Seasonality and surges are shared by all ward types of a kind | One ward type's year is too noisy to estimate its own | Ward types with their own patterns (for example, a winter-heavy respiratory-related ward type) would need more |
| Surge persistence from one year is rough | A year contains only a dozen or so independent surges | Bed numbers move by a few beds with it; the data bootstrap carries the uncertainty |
| Stays are resampled from the data year | Keeps the tail and each patient's flags | A stay longer than any seen can't occur; a long-stay tail fitted across several years would be steadier |
| Scenario flags are invented | No real data | The scenario savings illustrate the method, not likely savings |

---

## 11. Parameter reference

| Parameter | Value | Where |
|---|---|---|
| Ward types | 15 (10 emergency, 5 elective) | Synthetic hospital |
| Hospital-to-hospital variation | admissions ±15%, typical stay ±10%, long-stay share ×0.6–1.4, long-stay median ±15% | Synthetic hospital |
| Emergency season | ±10%, peak 15 January | Synthetic hospital |
| Elective season | August 75%, Christmas 40% | Synthetic hospital |
| Surges | AR(1), persistence 0.92, log-scale SD 0.10, emergency only | Synthetic hospital |
| Long stays | 0–6% of admissions, median 30–50 days, log-scale SD 0.45 | Synthetic hospital |
| Data year; extract taken | 1 April 2025 to 31 March 2026; 30 days after | Synthetic hospital |
| Core, escalation | 85%, 95% of nights | Both methods |
| Original: extract, fit, cap, warm-up | finished spells of 1–60 days; Tukey exclusion + five distributions by KS; 90 days; 30 days | Original |
| Recommended: stays, warm-up | all spells, Kaplan–Meier weights, resampled; 365 days | Recommended |
| Simulated years | 500 (dashboard), 1,000 (scripts) | Both |
| Data bootstrap | 30 copies, weeks redrawn from within 4 weeks, plus the year's surge level; 200 simulated years each | Recommended |
| Projection | admissions +1.5% a year, stays −1% a year | Both |

---

## 12. Next steps

1. **What happens when beds run out.** Add boarding on other wards and cancelled elective admissions, so the model can say what a given number of beds costs, not only how many avoid the problem.
2. **More than one year.** Fit the long-stay tail and the surges to several years, with a trend, rather than to one.
3. **A Bayesian length-of-stay model.** Partial pooling of each ward type's tail towards the others, which would steady the small ward types (as in the theatre scheduling project's duration model).
4. **Real data.** Run the same comparison on a real extract: the method's choices can be tested there too, even though the truth is unknown, by holding out a year.
