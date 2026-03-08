import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Load the data
data_path = Path("../../../FBSI/annotations.csv")
df = pd.read_csv(data_path)

print("=== Exploratory Data Analysis: Feed Bunk Score Annotations ===\n")

# Basic information
print("1. Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nData types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())

# Missing values
print("\n2. Missing Values Analysis:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found")

# Target variable analysis
print("\n3. Target Variable (Score) Analysis:")
print(f"Score distribution:\n{df['score'].value_counts().sort_index()}")
print(f"Score range: {df['score'].min()} - {df['score'].max()}")

# Categorical variables analysis
categorical_cols = ['farm', 'diet_composition', 'feed_bunk_background', 'feed_bunk_format', 'adjustment']

print("\n4. Categorical Variables Analysis:")
for col in categorical_cols:
    print(f"\n{col.upper()}:")
    print(f"Unique values: {df[col].nunique()}")
    print(f"Most common: {df[col].value_counts().head(3).to_dict()}")

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Feed Bunk Score Dataset Analysis', fontsize=16)

# Score distribution
sns.countplot(data=df, x='score', ax=axes[0,0])
axes[0,0].set_title('Score Distribution')
axes[0,0].set_xlabel('Score')
axes[0,0].set_ylabel('Count')

# Farm distribution
sns.countplot(data=df, y='farm', ax=axes[0,1], order=df['farm'].value_counts().index)
axes[0,1].set_title('Farm Distribution')
axes[0,1].set_xlabel('Count')
axes[0,1].set_ylabel('Farm')

# Feed bunk background
sns.countplot(data=df, y='feed_bunk_background', ax=axes[0,2], order=df['feed_bunk_background'].value_counts().index)
axes[0,2].set_title('Feed Bunk Background')
axes[0,2].set_xlabel('Count')
axes[0,2].set_ylabel('Background')

# Feed bunk format
sns.countplot(data=df, y='feed_bunk_format', ax=axes[1,0], order=df['feed_bunk_format'].value_counts().index)
axes[1,0].set_title('Feed Bunk Format')
axes[1,0].set_xlabel('Count')
axes[1,0].set_ylabel('Format')

# Adjustment
sns.countplot(data=df, x='adjustment', ax=axes[1,1])
axes[1,1].set_title('Adjustment Distribution')
axes[1,1].set_xlabel('Adjustment')
axes[1,1].set_ylabel('Count')

# Score vs Farm
score_farm = pd.crosstab(df['farm'], df['score'], normalize='index') * 100
score_farm.plot(kind='bar', stacked=True, ax=axes[1,2])
axes[1,2].set_title('Score Distribution by Farm (%)')
axes[1,2].set_xlabel('Farm')
axes[1,2].set_ylabel('Percentage')
axes[1,2].legend(title='Score', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('../../outputs/plots/eda_overview.png', dpi=300, bbox_inches='tight')
# plt.show()  # Commented out for headless environment

# Cross-tabulations
print("\n5. Cross-tabulations:")

# Score vs categorical variables
for col in categorical_cols[:-1]:  # Exclude adjustment for now
    print(f"\nScore vs {col}:")
    crosstab = pd.crosstab(df[col], df['score'])
    print(crosstab)

    # Chi-square test would be good here, but let's keep it simple
    print(f"Most common score for each {col}:")
    for val in df[col].unique():
        subset = df[df[col] == val]
        mode_score = subset['score'].mode().iloc[0]
        print(f"  {val}: {mode_score}")

# Diet composition analysis (seems complex)
print("\n6. Diet Composition Analysis:")
diet_counts = df['diet_composition'].value_counts()
print(f"Top 10 diet compositions:\n{diet_counts.head(10)}")

# Score by diet composition
diet_score = df.groupby('diet_composition')['score'].agg(['count', 'mean', 'std']).round(2)
diet_score = diet_score[diet_score['count'] > 5].sort_values('mean', ascending=False)
print(f"\nScore statistics by diet composition (min 5 samples):\n{diet_score.head(10)}")

# Create additional plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Score vs feed bunk background
background_score = pd.crosstab(df['feed_bunk_background'], df['score'], normalize='index') * 100
background_score.plot(kind='bar', stacked=True, ax=axes[0])
axes[0].set_title('Score Distribution by Feed Bunk Background (%)')
axes[0].set_xlabel('Background')
axes[0].set_ylabel('Percentage')
axes[0].legend(title='Score', bbox_to_anchor=(1.05, 1), loc='upper left')

# Score vs feed bunk format
format_score = pd.crosstab(df['feed_bunk_format'], df['score'], normalize='index') * 100
format_score.plot(kind='bar', stacked=True, ax=axes[1])
axes[1].set_title('Score Distribution by Feed Bunk Format (%)')
axes[1].set_xlabel('Format')
axes[1].set_ylabel('Percentage')
axes[1].legend(title='Score', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('../../outputs/plots/eda_background_format.png', dpi=300, bbox_inches='tight')
# plt.show()  # Commented out for headless environment

print("\n7. Key Insights:")
print("- Dataset contains", len(df), "samples")
print("- Score ranges from", df['score'].min(), "to", df['score'].max())
print("- Most common score:", df['score'].mode().iloc[0])
print("- Farms with most samples:", df['farm'].value_counts().head(3).index.tolist())
print("- Most common feed bunk background:", df['feed_bunk_background'].mode().iloc[0])
print("- Most common feed bunk format:", df['feed_bunk_format'].mode().iloc[0])

print("\nEDA completed. Plots saved to ../../outputs/plots/")
# add a figure with example images from the dataset with different scores, backgrounds, and formats if possible

