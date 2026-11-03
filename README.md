# Student Pass/Fail Clustering (Feature Engineering & K-Means)

**Student:** Ny Rosa  
**Course:** Machine Learning / Unsupervised Learning  
**Assignment:** Homework 02 - Student Pass/Fail Clustering with Feature Engineering  

---

## Overview

This project tests whether feature engineering can improve unsupervised K-Means clustering ($k=2$, `random_state=42`) on student academic records (`students_dataset.csv`, 600 students).

The goal is to separate students into Pass and Fail groups based only on behavioral features, keeping the `actual` label aside strictly to evaluate final accuracy.

---

## Workflow Followed

1. **Target Isolation (No Answer Leakage):**
   Immediately popped the `actual` column before doing any feature creation or clustering (`y = df.pop("actual")`). The unsupervised algorithm never sees the ground truth.

2. **Data Cleaning & Boundary Checks:**
   - Checked duplicates: `0` found.
   - Checked missing values: `0` found.
   - Checked ranges: percentages (`attendance_pct`, `assignments_done_pct`, `previous_grade`) are within `0–100%`, and hours/absences are non-negative.

3. **Feature Engineering:**
   - `sleep_deviation = abs(sleep_hours - 8)`: measures sleep disruption from the normal 8-hour baseline.
   - `risk_count = (study < 2) + (att < 70) + (grade < 50) + (abs >= 10)`: counts how many academic red flags a student has (0 to 4).
   - `engagement_score = (attendance_pct + assignments_done_pct) / 2`: candidate average of attendance and assignments.

4. **Multicollinearity Check (|r| > 0.85):**
   - `assignments_done_pct` and `engagement_score` had a correlation of **0.918**. Because it strongly duplicates the homework column, `engagement_score` was removed.
   - `sleep_hours` and `sleep_deviation` had a correlation of **-0.815** (below 0.85), so both were kept.
   - This leaves **8 final features** for clustering.

5. **Scaling & K-Means Clustering:**
   - Scaled all features with `StandardScaler()` so large scales like attendance (50–100) don't dominate small scales like sleep (4–10).
   - Trained K-Means ($k=2$, `random_state=42`) before and after feature engineering.

---

## Results Summary

| Metric | Before Feature Engineering | After Feature Engineering | Difference |
| :--- | :---: | :---: | :---: |
| **Features Used** | 6 original features | 8 features (+2 engineered) | +2 features |
| **Clustering Accuracy** | **82.17%** (493 / 600) | **84.17%** (505 / 600) | **+2.00% (+12 students)** |
| **Silhouette Score** | **0.295** | **0.294** | -0.001 (same) |

### Confusion Matrix (After Feature Engineering)

| | Predicted Fail | Predicted Pass | Total |
| :--- | :---: | :---: | :---: |
| **Actually Fail** | **169** | 41 | 210 |
| **Actually Pass** | 54 | **336** | 390 |
| **Total** | 223 | 377 | 600 |

*Overall Accuracy: (169 + 336) / 600 = 84.17%*

---

## Key Findings & Explanation

- **Did feature engineering help?**  
  Yes. Accuracy improved from 82.2% to 84.2%, meaning 12 additional students were grouped into their correct outcome. The silhouette score stayed essentially flat (0.295 vs 0.294), meaning cluster compactness in Euclidean space didn't change much, but the cluster boundary shifted closer to real-world passing/failing behavior.

- **Which feature helped most?**  
  `risk_count` helped the most. When tested by itself in our feature ablation test, adding `risk_count` alone reached **85.83% accuracy** (515/600 students). Combining multiple warning signs into a single count gives K-Means a very strong signal for struggling students.

- **Bonus (`np.log1p` on absences):**  
  Applying `np.log1p()` to `absences` dropped accuracy from 84.17% down to 82.50%. The skewness of `absences` is only 0.273 (already fairly symmetric), so taking the log unnecessarily compressed the gap between normal and chronically absent students.

---

## Repository Contents

- `pass_fail_clustering.ipynb`: Complete, executed Jupyter Notebook with all steps, metrics, and visual plots.
- `ANSWER_SHEET.md`: Completed answer sheet matching the assignment rubric.
- `students_dataset.csv`: Original dataset (600 rows, 7 columns).
- `students_engineered.csv`: Exported dataset with the final 8 engineered features (without `actual`).
- `Feature Engineering Criteria.pdf`: Official assignment prompt and grading criteria.
- `results/`:
  - `Answer_Sheet.docx`: Completed Word document submission.
  - `confusion_matrix_before_after.png`: Before vs After confusion matrix plot.
  - `silhouette_analysis.png`: Cluster silhouette diagrams.
  - `pca_clusters.png`: 2D PCA cluster projection.
  - `feature_ablation.png`: Bar chart comparing individual feature contributions.
  - `feature_ablation.csv`: Metrics table for each feature combination.
  - `correlation_matrix.png`: Heatmap showing correlation between features.
  - `metrics_comparison.csv`: Summary metrics comparison table.

---

## How to Run

Open `pass_fail_clustering.ipynb` in VS Code, JupyterLab, or Google Colab, and run the cells in order from top to bottom.
