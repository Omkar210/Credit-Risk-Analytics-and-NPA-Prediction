# Credit Risk Analytics & NPA Prediction

An end-to-end data analytics and predictive modeling project focused on credit risk assessment, data quality engineering, exploratory data analysis (EDA), feature engineering, and regression diagnostics for **Non-Performing Asset (NPA)** risk management. 

Using a massive dataset of **2,000,000 retail credit records** across 9 relational databases, this project models borrower behavior, identifies macro and micro risk drivers, and evaluates linear regression models to predict **Loss Given Default (LGD %)**.

---

## Project Structure & Workflow

The workflow of this project is structured into four main phases, as implemented in the master Jupyter Notebook [capstone-perform.ipynb](file:///c:/Users/Omkar/Downloads/Credit-Risk-Analytics-and-NPA-Prediction/capstone-perform.ipynb):

```mermaid
graph TD
    A[Data Integration & Downcasting] --> B[Data Cleaning & Imputation]
    B --> C[Exploratory Data Analysis]
    C --> D[Feature Engineering]
    D --> E[Multicollinearity Analysis VIF]
    E --> F[OLS Regression Modeling]
    F --> G[Regression Diagnostics & Recommendations]
```

---

## 1. Data Acquisition, Integration & Cleaning

### Data Scale & Memory Optimization
The project integrates credit data from 9 separate relational files containing borrower profile, credit bureau, asset collateral, payment history, and regional economic factors:
- `loans-master.csv`
- `customer-bureau.csv`
- `loan-performance.csv`
- `payment-history.csv`
- `monthly-emi-track.csv`
- `loan-enquiry-bureau.csv`
- `collateral-assets.csv`
- `credit-card-behavior.csv`
- `branch-region-economy.csv`

To handle the scale of **2,000,000 rows** efficiently, the data integration pipeline downcasts data types (e.g., converting `int64` to `int32`/`int16` and `float64` to `float32`/`float16`). 
- **Master Joined Data Shape**: 2,000,000 rows, 182 columns
- **Orphan Records**: **0** (perfect key alignment across all relational files)
- **Parquet Storage Optimization**: The consolidated file is saved as `master-dataframe.parquet`, reducing disk storage to **417.81 MB** with accelerated load times.

---

### Cleaning Deliberate Data Quality Issues
Four significant deliberate anomalies in the source data were detected and programmatically resolved:

| Identified Data Quality Issue | Diagnosis | Resolution Action | Affected Records |
| :--- | :--- | :--- | :---: |
| **NPA Status Mismatch** | Defaulted loans (`loan-status == 1`) mistakenly flagged as performing (`npa-flag == 0`). | Forced `npa-flag = 1` for all defaulted loans. Set `dirty-flag = 1`. | 77,541 |
| **Zero Collateral Value** | Secured loans (`has-collateral == 1`) recorded with a collateral value of `0` INR. | Imputed collateral value with the median of valid positive secured assets. | 120,084 |
| **Invalid Rejection Rate** | Rejection rates exceeding physical limits (`rejection-rate-pct > 100%`). | Capped the invalid values at exactly `100%`. | 3,168 |
| **Age & Credit Hist. Mismatch** | Credit history duration longer than the borrower's age (`credit-hist-years > age`). | Capped credit history length at `age - 18` (min. 0). Set `dirty-flag = 1`. | 132,333 |

---

### Classification-Based Missing Value Imputation
Missing values in high-missingness columns were classified by their missingness mechanisms and imputed systematically:

*   **Months Since Last Delinquency (`mths-since-last-delinq`)** — *1,098,640 missing records*
    *   **Mechanism**: **MNAR** (Missing Not at Random). A missing value indicates that the customer has never experienced a delinquency.
    *   **Imputation**: Imputed with a constant **`-1`** and tracked with a new binary indicator column `mths-since-last-delinq-missing = 1`.
*   **Mortgage Accounts (`mort-acc`)** — *260,330 missing records*
    *   **Mechanism**: **MAR** (Missing At Random).
    *   **Imputation**: Imputed using the column **median**.
*   **Employment Length in Years (`emp-length-years`)** — *180,539 missing records*
    *   **Mechanism**: **MAR** (Missing At Random).
    *   **Imputation**: Imputed using the column **median**.
*   **Installment Utility Percentage (`il-util-pct`)** — *200,167 missing records*
    *   **Mechanism**: **MAR** (Missing At Random).
    *   **Imputation**: Imputed using the column **median**.

---

### Outlier Treatment via Winsorization
To prevent extreme values from distorting predictive regression models, the 6 most highly skewed numeric columns were Winsorized by clipping values at their **1st** and **99th** percentiles:

1.  **`collections-12mths-fee`** (Skew: 139.04) → Capped at `0.0` (p1) and `657.22` (p99)
2.  **`collection-recovery-fee`** (Skew: 125.77) → Capped at `0.0` (p1) and `2910.98` (p99)
3.  **`recoveries-inr`** (Skew: 94.23) → Capped at `0.0` (p1) and `40329.10` (p99)
4.  **`emi-advance-paid-inr`** (Skew: 59.17) → Capped at `0.0` (p1) and `40329.10` (p99)
5.  **`expected-loss-inr`** (Skew: 27.01) → Capped at `0.0` (p1) and `3990.42` (p99)
6.  **`avg-cur-bal-inr`** (Skew: 25.66) → Capped at `47.52` (p1) and `44732.77` (p99)

---

## 2. Exploratory Data Analysis (EDA) Insights

*   **Class Imbalance**: Out of 2,000,000 loans in the master dataset, **96.1% are performing** (1,922,444 loans) and **3.9% are defaulted/NPAs** (77,556 loans). 
*   **CIBIL Distribution**: Performing borrowers show a distribution skewed heavily toward high credit scores (700+), while defaulted borrowers exhibit a flattened, broader distribution concentrated at lower scores.
*   **Key Risk Drivers**: Boxplots show that defaulted loans systematically feature:
    *   Higher Interest Rates (`int-rate-pct`)
    *   Higher Debt-to-Income (`dti-pct`) ratios
    *   Lower Annual Incomes (`annual-inc-inr`)
    *   Higher credit utilization ratios (`revol-util-pct`)
*   **Credit Quality Grading**: Default rates climb monotonically across credit grades. Grade A loans exhibit the lowest default rate (<1%), while Grade G loans show a default rate exceeding 15%.
*   **Temporal & Macroeconomic Trends**:
    *   Default rates spiked significantly in **2020** to **1.79%** (compared to a non-COVID average of **1.54%**) due to pandemic-related disruptions.
    *   A dual-axis overlay of the RBI Repo Rate and default rates demonstrates that high-rate environments increase the repayment burden, leading to an increase in defaults with a short temporal lag.

> [!NOTE]
> **The Loss Given Default (LGD) Paradox:**
> A crucial statistical finding is that CIBIL score is heavily correlated with whether a customer defaults (Probability of Default - PD), but has **zero correlation with Loss Given Default (LGD %)** on defaulted loans (**r = 0.0018**, p-value = 0.623). Once a default occurs, credit bureau history does not predict what percentage of the asset the bank will lose. LGD is instead driven by collateral liquidation, recovery policies, and macroeconomics.

---

## 3. Feature Engineering

To capture non-linearities and credit relationships, 13 complex features were engineered and evaluated against LGD:

### A. Repayment-Burden Features
*   **`emi-to-income-ratio`**: Monthly installment relative to monthly income.
    $$\text{EMI to Income Ratio} = \frac{\text{installment\-inr}}{\text{annual\-inc\-inr} / 12}$$
*   **`loan-to-income-ratio`**: Total loan amount relative to annual income.
    $$\text{Loan to Income Ratio} = \frac{\text{loan\-amnt\-inr}}{\text{annual\-inc\-inr}}$$
*   **`rate-spread-pct`**: Spread over the central bank policy rate (Strongest correlation with LGD: **0.0795**).
    $$\text{Rate Spread \%} = \text{int\-rate\-pct} - \text{rbi\-repo\-rate\-pct}$$
*   **`real-interest-rate`**: Real interest rate adjusted for CPI inflation.
    $$\text{Real Interest Rate} = \text{int\-rate\-pct} - \text{cpi\-inflation\-pct}$$

### B. Bureau-Behavior Features
*   **`credit-util-composite`**: A weighted credit utilization index focusing on revolving and bankcard accounts.
    $$\text{Credit Util Composite} = 0.5 \times \text{revol\-util\-pct} + 0.3 \times \text{bc\-util\-pct} + 0.2 \times \text{all\-util\-pct}$$
*   **`delinq-severity-score`**: Historical delinquency rate weighted by recency.
    $$\text{Delinq Severity} = \text{delinq\-2yrs} \times \left(1 + \frac{1}{\max(1, \text{mths\-since\-last\-delinq})}\right)$$
*   **`enq-velocity-score`**: Inquiry frequency heavily weighting recent 30-day activity.
    $$\text{Enquiry Velocity} = \text{num\-enquiries\-30d} \times 4 + \text{num\-enquiries\-90d}$$

### C. Income and Collateral Features
*   **`income-stability-ratio`**: Income adjusted for job tenure.
    $$\text{Income Stability} = \frac{\text{annual\-inc\-inr}}{\text{emp\-length\-years} + 1}$$
*   **`credit-depth-score`**: Total credit accounts relative to history length.
    $$\text{Credit Depth} = \frac{\text{total\-acc}}{\text{credit\-hist\-years} + 1}$$
*   **`collateral-coverage-ratio`**: Asset coverage relative to loan amount.
    $$\text{Collateral Coverage} = \frac{\text{collateral\-value\-inr}}{\text{loan\-amnt\-inr} + 1}$$

### D. Skew Reduction & Macro Context
*   **`log-annual-inc`** & **`log-loan-amnt`**: Log-transformed financial variables to stabilize variance.
*   **`covid-issue-year-flag`**: Binary flag indicating whether the loan was issued in 2020. An independent t-test verified that COVID-era defaulted loans had a significantly higher mean LGD (**1.79%** vs **1.54%** overall, **t-statistic: 10.63, p-value: 0.0000**).

---

## 4. Regression Modeling & Diagnostics

### Multicollinearity Analysis
To satisfy OLS assumptions, Variance Inflation Factors (VIF) were calculated across candidate features. An initial model showed severe multicollinearity. Four variables were dropped to resolve this:

*   **Dropped Variables**: `rate-spread-pct` (VIF: 7.82), `log-annual-inc` (VIF: 2.44), `log-loan-amnt` (VIF: 2.34), `open-acc` (VIF: 2.35).
*   **Result**: All retained features successfully dropped to VIF levels **< 6.0** (with highest being `emi-to-income-ratio` at 5.74).

#### VIF Comparison Table

| Feature | Initial VIF | Retained VIF |
| :--- | :---: | :---: |
| **`const`** | 912.66 | 137.93 |
| **`cibil-score`** | 1.00 | 1.00 |
| **`age`** | 1.03 | 1.03 |
| **`dti-pct`** | 1.00 | 1.00 |
| **`total-acc`** | 2.61 | 1.26 |
| **`emp-length-years`** | 1.18 | 1.12 |
| **`ltv-ratio-pct`** | 1.40 | 1.28 |
| **`collateral-score`** | 1.00 | 1.00 |
| **`loan-term-months`** | 1.28 | 1.28 |
| **`emi-to-income-ratio`** | 5.74 | 5.74 |
| **`loan-to-income-ratio`** | 7.09 | 5.66 |
| **`real-interest-rate`** | 7.71 | 1.01 |
| **`credit-util-composite`** | 1.00 | 1.00 |
| **`delinq-severity-score`** | 1.00 | 1.00 |
| **`enq-velocity-score`** | 1.00 | 1.00 |
| **`income-stability-ratio`** | 1.72 | 1.19 |
| **`credit-depth-score`** | 1.30 | 1.30 |
| **`collateral-coverage-ratio`** | 1.28 | 1.17 |
| **`covid-issue-year-flag`** | 1.15 | 1.00 |
| *`rate-spread-pct`* | *7.82* | *Dropped* |
| *`log-annual-inc`* | *2.44* | *Dropped* |
| *`log-loan-amnt`* | *2.34* | *Dropped* |
| *`open-acc`* | *2.35* | *Dropped* |

---

### Baseline OLS Regression Model & Diagnostics
An Ordinary Least Squares (OLS) regression model was fitted on defaulted loans ($N = 77,556$) to predict `lgd-pct`:

```
                                 OLS Regression Results
==============================================================================
Dep. Variable:                lgd-pct   R-squared:                       0.000
Model:                            OLS   Adj. R-squared:                 -0.000
Method:                 Least Squares   F-statistic:                    0.9043
Prob (F-statistic):             0.590   Log-Likelihood:            -3.4231e+05
No. Observations:               77556   AIC:                         6.847e+05
Df Residuals:                   77533   BIC:                         6.849e+05
Df Model:                          22   
==============================================================================
```

> [!WARNING]
> **Key Modeling Insight:**
> The OLS model achieved an **R-squared of 0.000** and an F-statistic p-value of **0.590** (statistically insignificant). This indicates that **linear models of borrower risk metrics are ineffective for predicting LGD**. 
> While bureau and financial profile features can predict the *occurrence* of default (PD), they do not capture the post-default recovery rates (LGD) which are driven by asset liquidation efficiency, collection agency quality, and recovery duration.

#### OLS Diagnostic Indicators
*   **Residual Normality**: The Omnibus statistic (**4,458.99**, p-value = 0.00) and Jarque-Bera statistic (**2,494.49**, p-value = 0.00) indicate that residuals are highly non-normal.
*   **Autocorrelation**: The Durbin-Watson statistic is **1.995**, indicating that residuals do not show significant autocorrelation.
*   **Numerical Instability**: The high Condition Number (**6.35e+06**) suggests remaining scaling issues (e.g., mixing raw currency values in INR with ratio and percentage metrics), indicating OLS is numerically unstable.

---

## NPA Management Recommendations

1.  **Decouple PD and LGD Modeling**: Do not use credit bureau scores (e.g. CIBIL) to evaluate post-default losses. CIBIL is excellent for predicting PD, but has no linear relation to LGD.
2.  **Focus on Collateral Valuations**: Shift focus to high-fidelity collateral valuation, legal recovery channels, and asset-backed structural features (like LTV ratio and collateral coverage) when modeling LGD.
3.  **Deploy Non-Linear Models**: OLS struggles with LGD due to extreme values and zero-inflation (many defaults yield either 0% or 100% loss). Advanced models like Two-Stage (Tobit) models, Beta Regressions, or Tree-Based ensemble algorithms (Random Forest, XGBoost) should be used instead of standard linear regression.
