import matplotlib.pyplot as plt
import seaborn as sns
from data_cleaning_3 import df_clean
from feature_transformation_5 import df_transformed
from data_reduction_pca_6 import df_pca_components

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Outlier Check (Boxplot)
sns.boxplot(data=df_clean[["MonthlyCharges", "TotalCharges"]], ax=axes[0, 0])
axes[0, 0].set_title("Pre-Transformation Boxplots")

# 2. Feature Distribution (Histogram)
sns.histplot(df_clean["MonthlyCharges"], kde=True, ax=axes[0, 1], color='teal')
axes[0, 1].set_title("Monthly Charges Distribution")

# 3. Binned Tenure Distribution
sns.countplot(x=df_clean["tenure"], ax=axes[1, 0], palette="Set2")
axes[1, 0].set_title("Tenure Spread")

# 4. PCA Component Scatter Plot
sns.scatterplot(x="PC1", y="PC2", data=df_pca_components, ax=axes[1, 1], alpha=0.6)
axes[1, 1].set_title("PCA Feature Projection (2 Components)")

plt.tight_layout()
plt.show()