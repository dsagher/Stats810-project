#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-12-09T21:57:16.585Z
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("smoking.csv")
df = df.iloc[:, 1:]

df.isna().sum()

# Above is a check to see if there are any missing values in the smoking dataset, which there appear to be none.


df['smoking'].value_counts()
df['smoking'].value_counts(normalize=True)
sns.countplot(data=df, x='smoking')
plt.title("Smoking Status Distribution")
plt.show()

# There appears to be a good amount more nonsmokers than smokers in the dataset, with the ratio being somewhere near 4:7 smoker to nonsmoker.


df_corr = df.copy()

# Convert smoking to numeric if it's not already (e.g., Yes/No -> 1/0)
df_corr['smoking'] = df_corr['smoking'].astype(int)

# All continuous variables + smoking
corr_vars = [
 'age','height(cm)','weight(kg)','waist(cm)',
 'systolic','relaxation','fasting blood sugar',
 'Cholesterol','triglyceride','HDL','LDL',
 'hemoglobin','serum creatinine','AST','ALT','Gtp',
 'smoking'
]

plt.figure(figsize=(14,10))
sns.heatmap(df_corr[corr_vars].corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap (Including Smoking)")
plt.show()

# AST and ALT appear to have strong positive correlations with one another, HDL being negatively correlated with waist size and weight. While not amazingly strong, smoking seems to have the highest positive correlations with hemoglobin, weight a bit less so, while negative correlations seem to come with age, HDL, and LDL.


continuous = [
 'age','height(cm)','weight(kg)','waist(cm)',
 'systolic','relaxation','fasting blood sugar',
 'Cholesterol','triglyceride','HDL','LDL',
 'hemoglobin','serum creatinine','AST','ALT','Gtp'
]
df[continuous].hist(bins=30, figsize=(14,10))
plt.suptitle("Histograms of Continuous Variables")
plt.show()
plt.tight_layout()

# Continuous variables are checked with the 4x4 matrix of histograms, some of which like hemoglobin and waist size seem to be pretty normally distributed, while a lot, like AST, ALT, and LDL seem to have a pretty skewed right distribution, indicating perhaps some outliers.


key_vars = ['triglyceride', 'HDL', 'LDL', 'Gtp', 'AST', 'ALT', 'fasting blood sugar']

plt.figure(figsize=(16, 18))

for i, col in enumerate(key_vars, 1):
    plt.subplot(4, 2, i)
    sns.violinplot(data=df, x='smoking', y=col, inner=None)
    sns.boxplot(data=df, x='smoking', y=col, width=0.2, showcaps=False, boxprops={'facecolor':'white'})
    plt.title(f"{col} Distribution by Smoking Status")

plt.tight_layout()
plt.show()

# The violin plots compare smokers (1) against nonsmokers (0) across variables such as triglyceride, HDL, AST, and fasting blood sugar. There seem to be a good amount of outliers, which backs up the assertion made about the histogram matrix and its plots.