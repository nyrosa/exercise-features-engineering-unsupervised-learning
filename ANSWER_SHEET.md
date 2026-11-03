# Answer Sheet: Feature Engineering for Pass/Fail Clustering

**Course:** Machine Learning & Unsupervised Learning  
**Assignment:** Student Pass/Fail Clustering with Feature Engineering  

---

### Student Details
| Field | Details |
| :--- | :--- |
| **Name** | Ny Rosa |
| **Date submitted** | 2nd Nov 2026 |
| **Dataset** | `students_dataset.csv` (600 students) |

---

### Part A: Data Cleaning (20 points)

| Check | Code you used | Result (number) | What you did |
| :--- | :--- | :--- | :--- |
| **Number of rows and columns** | `df.shape` | `(600, 7)` | 600 students: 6 feature columns + 1 target (`actual`) |
| **Duplicate rows** | `df.duplicated().sum()` | `0` | Nothing to remove; no duplicate rows found. |
| **Empty cells** | `df.isna().sum().sum()` | `0` | Nothing to fill; dataset contains zero missing values. |
| **Values out of range (e.g. attendance > 100)** | `((df[cols] < 0) \| (df[cols] > 100)).sum()` | `0` | All values are valid within expected boundaries [0, 100]. |
| **Old prediction columns found? (Yes / No)** | `df.columns` | `No` | Only the 6 features and the true outcome `actual`. |

#### Question A6
**At which step did you remove the `actual` column from the features?**  
> **Answer:**  
> Right after loading the data, before performing any data cleaning, feature engineering, or clustering:  
> ```python
> y = df.pop("actual")
> ```  
> This isolates the ground truth completely, preventing **data leakage** and guaranteeing that the unsupervised K-Means algorithm never peeks at the target.

---

### Part B: New Features (30 points)

| # | Feature name | Formula / Code | Why it might affect passing (one sentence) |
| :---: | :--- | :--- | :--- |
| **1** | `sleep_deviation` | `(df["sleep_hours"] - 8).abs()` | Too little or too much sleep lowers concentration during study and exams. |
| **2** | `risk_count` | `(df.study_hours_per_week < 2).astype(int) + (df.attendance_pct < 70) + (df.previous_grade < 50) + (df.absences >= 10)` | The more warning signs a student has, the more likely they are to fail. |
| **3** *(optional)* | `engagement_score` | `(df.attendance_pct + df.assignments_done_pct) / 2` | Students who attend class and submit homework are better prepared. *(Removed in Part C)* |

---

### Part C: Remove Repeated Features and Scale (10 points)

| Feature 1 | Feature 2 | Correlation ($r$) | Which one did you remove? |
| :--- | :--- | :---: | :--- |
| `assignments_done_pct` | `engagement_score` | `0.918` | **`engagement_score`** (it mostly repeats `assignments_done_pct`, exceeding the 0.85 threshold). |
| `sleep_hours` | `sleep_deviation` | `-0.815` | **None:** $|r| = 0.815 < 0.85$, so both are kept. |

#### Question C1
**Final list of features used for K-Means:**  
> **Answer:**  
> `study_hours_per_week`, `attendance_pct`, `previous_grade`, `assignments_done_pct`, `sleep_hours`, `absences`, `sleep_deviation`, `risk_count` (**8 features total**).

#### Question C2
**Did you scale the features with `StandardScaler()`?**  
> **Answer:**  
> **Yes.** K-Means relies entirely on Euclidean distances ($\sqrt{\sum (x_i - y_i)^2}$). Without scaling, high-range features like `attendance_pct` ($0 \dots 100$) would dominate low-range features like `sleep_hours` ($4 \dots 10$). `StandardScaler()` gives all features zero mean and unit variance.

---

### Part D: Results (20 points)

| Metric | Before Feature Engineering | After Feature Engineering | Better or worse? |
| :--- | :---: | :---: | :--- |
| **Accuracy (compared with `actual`)** | `0.822` *(493 / 600)* | `0.842` *(505 / 600)* | **Better** (+0.020, **12 more students** correct) |
| **Silhouette score** | `0.295` | `0.294` | **About the same** (−0.001) |

#### Confusion Matrix (After Feature Engineering)

| | Predicted Fail | Predicted Pass | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actually Fail** | **169** | 41 | 210 |
| **Actually Pass** | 54 | **336** | 390 |
| **Total Predicted** | 223 | 377 | 600 |

*(Overall accuracy: $(169 + 336) / 600 = 505 / 600 = \mathbf{84.17\%}$)*

---

### Part E: Explanation (20 points)

#### Question E1: Did your feature engineering improve the result?
> **Answer:**  
> Yes, slightly. Accuracy rose from **0.822 to 0.842**, meaning **12 additional students** were grouped correctly into their true academic outcomes. The silhouette score stayed essentially identical (**0.295 $\rightarrow$ 0.294**), indicating that while the geometrical compactness and distance between clusters remained the same, the decision boundary shifted to align much more accurately with student pass/fail reality.

#### Question E2: Which new feature helped most, and why?
> **Answer:**  
> **`risk_count`** helped the most. It aggregates four individual academic red flags (insufficient study time, low attendance, failing prior grade, high absences) into a single composite risk indicator, making struggling students distinct in the multidimensional space. In contrast, `sleep_deviation` alone provides weaker separation, but `risk_count` provides a strong, high-signal discriminator.
> 
> **Empirical Feature Ablation Study:**
> | Feature Combination | Features Used | Accuracy | Silhouette | Correct Students |
> | :--- | :---: | :---: | :---: | :---: |
> | **Original 6 features** | 6 | `0.822` | `0.295` | 493 / 600 |
> | **+ `risk_count` only** | 7 | **`0.858`** | **`0.312`** | **515 / 600** *(Highest!)* |
> | **+ `sleep_deviation` only** | 7 | `0.818` | `0.275` | 491 / 600 |
> | **+ `sleep_deviation` + `risk_count` (Final 8)** | 8 | `0.842` | `0.294` | 505 / 600 |
> | **+ All 3 (keeps collinear `engagement_score`)** | 9 | `0.828` | `0.297` | 497 / 600 |
> 
> Adding `risk_count` alone yields the single highest clustering accuracy (**85.83%**, correctly grouping 515 out of 600 students). When combined with `sleep_deviation`, it still yields a solid **84.17%** (+12 students over the original baseline). Keeping the redundant `engagement_score` degrades accuracy to 82.83%, validating why we removed it in Step 4.

#### Question E3: Why must `actual` not be used as a feature?
> **Answer:**  
> Because `actual` is the target ground truth we want the model to group without supervision. If `actual` is left in the feature set, K-Means would simply group students by that column directly (creating artificial, inflated 100% accuracy). This is severe **data leakage** and produces a useless model that cannot generalize to new students whose outcomes are unknown.

---

### Bonus (Optional): Skewness & Log Transformation (`np.log1p()`)

| Test | Accuracy | Silhouette Score | Finding |
| :--- | :---: | :---: | :--- |
| **Without `np.log1p(absences)`** | `0.842` | `0.294` | Baseline engineered model |
| **With `np.log1p(absences)`** | `0.825` | `0.274` | Decreased accuracy by −0.017 |

> **Explanation:**  
> `absences` has a skewness of **0.273**, which is near-symmetric (well within the $-0.5 \dots +0.5$ range). Applying `np.log1p()` compressed the upper values of absences, which compressed the geometric distance between normal and chronically absent students, slightly degrading K-Means performance.
