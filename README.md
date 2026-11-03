# Student Pass/Fail Clustering: Feature Engineering & Unsupervised Learning

This project explores **Feature Engineering** and **Unsupervised Learning** (specifically **K-Means Clustering**) to group students into Pass/Fail categories based on their academic behavior and historical metrics.

---

## 🎯 Project Objectives

1. **Clean and Validate Data**: Handle missing values, duplicate entries, and boundary checks.
2. **Prevent Answer Leakage**: Strictly isolate the `actual` target column before feature creation or clustering.
3. **Domain Feature Engineering**: Create intuitive features (`sleep_deviation`, `risk_count`, `engagement_score`).
4. **Multicollinearity Pruning**: Detect features with correlation $|r| > 0.85$ and drop redundant ones (`engagement_score`).
5. **Feature Standardization**: Scale features with `StandardScaler()` so Euclidean distance operates fairly.
6. **K-Means Clustering ($k=2$)**: Train clustering before and after feature engineering and evaluate using:
   - **Silhouette Score** (intrinsic cluster separation)
   - **Accuracy & Confusion Matrix** (external validation against actual outcomes).
7. **Bonus Analysis**: Test logarithmic transformation (`np.log1p()`) on `absences`.

---

## 📊 Summary of Results

| Metric | Before Feature Engineering | After Feature Engineering | Net Change |
| :--- | :---: | :---: | :---: |
| **Features Used** | 6 original features | 8 features | +2 engineered |
| **Clustering Accuracy** | **0.822** (493 / 600) | **0.842** (505 / 600) | **+0.020 (+12 students)** |
| **Silhouette Score** | **0.295** | **0.294** | -0.001 (consistent) |

### Confusion Matrix (After Feature Engineering)
```
                Predicted Fail  Predicted Pass
Actually Fail        169             41
Actually Pass         54            336
```
- **Overall Accuracy**: $(169 + 336) / 600 = 84.17\%$

---

## 📁 Repository Structure

```
├── students_dataset.csv             # Raw input dataset (600 students, 7 columns)
├── students_engineered.csv          # Final engineered dataset (8 features, no answer leakage)
├── pass_fail_clustering.ipynb       # Complete, executed Jupyter Notebook with all outputs & plots
├── ANSWER_SHEET.md                  # Completed assignment answer sheet (Markdown format)
├── Feature Engineering Criteria.pdf # Assignment specification and grading rubric
└── results/
    ├── Answer_Sheet.docx            # Completed answer sheet (MS Word format)
    ├── confusion_matrix_before_after.png # Colab-style confusion matrix plot (Before vs After)
    ├── silhouette_analysis.png      # Cluster silhouette diagrams (Before vs After)
    ├── pca_clusters.png             # 2D PCA projection of student clusters
    ├── feature_ablation.png         # Bar chart comparing all feature combinations
    ├── feature_ablation.csv         # Table metrics for all feature combinations
    ├── correlation_matrix.png       # Correlation heatmap showing multicollinearity
    └── metrics_comparison.csv       # Summary metrics table
```

---

## 🚀 How to Run

Open [`pass_fail_clustering.ipynb`](file:///home/rosa/ai-exercise/exercise-4/pass_fail_clustering.ipynb) in **VS Code**, **JupyterLab**, or upload it to **Google Colab** to execute cells interactively.

---

## 💡 Key Learnings

1. **Why K-Means is Unsupervised**:
   K-Means does not use labels during training. It iteratively partitions points into $k$ groups by minimizing the within-cluster sum of squared distances to the nearest centroid.
2. **Why Scaling Matters**:
   Because K-Means uses Euclidean distance, a feature with values between 0–100 (like attendance) would exert 10× more influence than a feature with values between 4–10 (like sleep). Standardizing to mean 0 and variance 1 puts all features on equal footing.
3. **Why Remove Target (`actual`)**:
   Keeping the target in unsupervised clustering leads to trivial clustering (data leakage). The algorithm must discover patterns purely from behavioral signals.
4. **Why Feature Engineering Helps**:
   Features like `risk_count` combine multiple weak academic indicators into a strong combined signal, making high-risk students geographically isolated from passing students in high-dimensional space.
