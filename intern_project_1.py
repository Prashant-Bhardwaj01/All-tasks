import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("heart_cleveland_upload.csv")
print(df.head())
print(df.info())

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
# Histogram-> Age Distribution (cont. variable)
sns.histplot(data=df, x='age', bins=15, kde=True, ax=axs[0], color='skyblue')
axs[0].set_title('Age Distribution')
axs[0].set_xlabel('Age')
axs[0].set_ylabel('Count')
# Bar Chart -> Gender Distribution (categorical variable)
g_counts = df['gender'].value_counts()
sns.barplot(x=g_counts.index, y=g_counts.values, ax=axs[1], palette='Set2')
axs[1].set_title('Gender Distribution')
axs[1].set_xlabel('Gender')
axs[1].set_ylabel('Count')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Condition (target) distribution
sns.countplot(data=df, x='condition', palette='pastel', ax=axs[0])
axs[0].set_title('Heart Disease Condition Distribution (0=No, 1=Yes)')
axs[0].set_xlabel('Condition')
axs[0].set_ylabel('Count')
cp = df['cp'].value_counts().sort_index()  # sort_index to keep order 0-3
# Age vs. Cholesterol scatter plot colored by condition
sns.scatterplot(data=df, x='age', y='chol', hue='condition', palette='coolwarm', ax=axs[1])
axs[1].set_title('Age vs Cholesterol Colored by Heart Disease Condition')
axs[1].set_xlabel('Age')
axs[1].set_ylabel('Cholesterol')
# Chest Pain Type Distribution
plt.figure(figsize=(8, 5))
sns.lineplot(x=cp.index, y=cp.values, marker='o', color='red') 
plt.title('Chest Pain Type Distribution (Line Plot)', fontsize=14)
plt.xlabel('Chest Pain Type (cp)')
plt.ylabel('Count')
plt.grid(True)
plt.xticks(cp.index)
plt.show()
