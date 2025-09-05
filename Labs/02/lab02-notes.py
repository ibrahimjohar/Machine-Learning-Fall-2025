#EDA
#EDA is crucial because it helps:
#1. Understand the data – Know the structure, distribution, and key characteristics.
#2. Detect errors and anomalies – Identify missing values, outliers, or inconsistencies.
#3. Guide modeling – Suggest which features are important and what transformations may be required.
#4. Validate assumptions – Many statistical/ML methods assume normality, independence, or linearity.
#5. Communicate insights – Visualizations and summaries provide a story of the data for stakeholders.

"""
Steps Involved in EDA
Although approaches vary, a standard EDA workflow includes:
1. Data Collection & Import
    o Load data from sources (CSV, database, API, etc.).
    o Check file format and encoding issues.
2. Data Inspection
    o View data shape (rows, columns).
    o Understand variable types (categorical, numerical, datetime, text).
    o Preview sample rows to get a feel for the dataset.
3. Data Cleaning
    o Handle missing values (drop, impute, or flag).
    o Remove duplicates.
    o Correct data types (e.g., strings to dates).
    o Standardize units and categories.
4. Univariate Analysis (Single Variable)
    o Numerical: Distribution, summary stats (mean, median, std, quartiles).
    o Categorical: Frequency counts, bar plots.
    o Identify skewness, outliers, or anomalies.
5. Bivariate/Multivariate Analysis
    o Correlations between numerical variables.
    o Cross-tabulations for categorical variables.
    o Scatterplots, boxplots, and heatmaps to check relationships.
6. Outlier Detection
    o Use visualization (boxplots, histograms) or statistical rules (IQR, Z-score).
    o Decide whether to keep, cap, or remove.
7. Feature Engineering (Preliminary)
    o Create new variables (ratios, groupings, transformations).
    o Encode categorical variables (label encoding, one-hot encoding).
8. Visualization
    o Histograms, bar charts, scatter plots, pair plots, correlation heatmaps.
    o Helps identify trends, clusters, and anomalies.
9. Hypothesis Generation
    o Based on patterns, generate insights/questions for deeper analysis.
    o Example: “Sales drop in Q3 may be linked to seasonal effects.”
10. Reporting/Documentation
    o Summarize findings with plots, tables, and key observations.
    o Provides a foundation for modeling, dashboards, or further analysis.
"""
import pandas as pd

df = pd.DataFrame({
    'test_score':      [35, 40, 28, 45, 33, 38, 29, 42, 31, 36],
    'writing_skills':  [30, 50, 25, 60, 28, 45, 22, 65, 33, 48],
    'reading_skills':  [32, 42, 20, 55, 30, 40, 18, 58, 28, 43],
    'attendance':      [60, 65, 55, 70, 50, 68, 58, 75, 62, 66],
    'study_hours':     [1, 2, 1, 3, 2, 2, 1, 3, 1, 2],
    'pass':            ['No','No','No','Yes','No','No','No','Yes','No','No']
})


#METHODS OF DATA CLEANING
#1. Handling Missing Values

# identifying
df.isnull()         #shows True for missing cells
df.isnull().sum()   #total missing values per column 

# ways to handle values

# Drop (simple, but can be risky for bias)
#   drop ROWS with few non-missing fields (dropna(axis=0))
#   drop COLUMNS if largely missing and low-importance (dropna(axis=1))
 
df_drop_row = df.drop.na()          #drop ROWS with any missing value

df_drop_col = df.drop.na(axis=1)    #drop COLS with any missing values

# Simple Imputations (fast baslines)

# NUMERIC -> MEDIAN (when there are OUTLIERS) | MEAN (when there are NO OUTLIERS)

df['Age'].fillna(df['Age'].mean(), inplace=True)

df['Score'].fillna(df['Score'].median(), inplace=True)

# CATEGORICAL -> MODE or a special label like "Unknown"

df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)

# Time-series gaps

# FORWARD-FILL (ffill) -> propogate previous value
# BACK-FILL (bfill) -> propogate next value

df['Age'].ffill(inplace=True)   #Forward Fill
df['Age'].bfill(inplace=True)   #Backward Fill

# Model-based imputation

# predict missing using other features
# used for complex datasets

from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(strategy='mean')
df[['Age', 'Score']] = imp_mean.fit_transform(df[['Age', 'Score']])  

#2. Removing Duplicates

# Duplicate records occur when one or more rows in a dataset are repeated, either fully or partially.
# Duplicates can lead to:
#   • Incorrect analytics (overcounting)
#   • Biased model training
#   • Misleading visualizations

# detect complete duplicates
duplicates = df.duplicated()
print(f"duplicate flags (full row): {duplicates}")

# ways to handle duplicate records

# 1.drop FULL ROW duplicates

df_cleaned = df.drop_duplicates()
print(f"after dropping full row duplicates: {df_cleaned}")

# 2.drop duplicates based on SPECIFIC COLUMNS

df_subset = df.drop_duplicates(subset=['Name', 'Age'])
print(f"after dropping duplicates on ['Name', 'Age']: {df_subset}")


#3. Feature Scaling

# feature scaling is a preprocessing step used to normalize the range of independent features (input variables) in the dataset
#   -many ML models (like KNN, SVM, Logistic Regression, Gradient Descent-based models) are sensitive to the scale of features
#   -features with larger ranges may dominate those with smaller ranges, skewing the model's performace

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# common feature scaling techniques

#1. Min-Max Scaling (Normalization)
#   -rescales data to a fixed range: [0,1]
#   -formula: X_scaled = (X - X_min) / (X_max - X_min)

scaler = MinMaxScaler() #creates a Min-Max Scaler object from scikit-learn, which rescales values of each feature to a range between 0 and 1 (by default).
scaled_min = scaler.fit_transform(df) #fits the scaler to the data (df) and transforms it, i.e., applies the min-max scaling formula: X_scaled = (X - X_min) / (X_max - X_min) 

#the result (scaled_min) is a NumPy array containing the scaled values.

#converts the scaled NumPy array back into a pandas DataFrame, keeping the original column names for readability.
df_scaled_min = pd.DataFrame(scaled_min, columns=df.columns)
print(f"Min-Max Scaled Data: {df_scaled_min}")

#SAMPLE

#original dataframe
df = pd.DataFrame({
    "Age": [20,25, 30, 35, 40],
    "Salary": [20000, 30000, 40000, 50000, 60000]
})

print(f"original data: {df}")

"""
Output (original data):
        Age  Salary
    0   20   20000
    1   25   30000
    2   30   40000
    3   35   50000
    4   40   60000
"""

#apply Min-Max Scaling
scaler = MinMaxScaler()
scaled_min = scaler.fit_transform(df)
df_scaled_min = pd.DataFrame(scaled_min, columns=df.columns)

print(f"Min-Max Scaled Data: {df_scaled_min}")

"""
Output (Min-Max Scaled Data - Range: [0-1])
        Age    Salary
    0   0.00    0.00
    1   0.25    0.25
    2   0.50    0.50
    3   0.75    0.75
    4   1.00    1.00
"""
#2. Standard Scaling (Z-score Normalization)
#   -transforms data to have MEAN = 0 & STANDARD DEVIATION = 1
#   -formula: X_scaled = (X - μ) / (σ)

scaler = StandardScaler()
scaled_standard = scaler.fit_transform(df)

df_standard = pd.DataFrame(scaled_standard, columns=df.columns)
print(f"standard scaled data: {df_standard}")

#4. Outliers Treatment

# outliers are data points that significantly differ from other observations, they can:
#   -indicate variability in data
#   -be the result of measurement errors, or
#   -represent rare events (e.g., fraud, failure)

# common detection techniques

#1. Z-score (standard score method)
#   -assumes normal distribution
#   -formula: Z = (x - μ) / (σ)
#   -flag if |Z| > 3 (common threshold)

from scipy.stats import zscore
import pandas as pd

df = pd.DataFrame({'Salary': [30000, 32000, 31000, 30500, 90000]})
df['Z'] = zscore(df['Salary'])
outliers = df[df['Z'].abs() > 3]
print(outliers)

#2. using plots (boxplot, scatterplot, histograms)
#   - boxplot - for univariate outlier detection
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.boxplot(x=df['Age'])
plt.title('Boxplot of Age')
plt.show()

# points outside the whiskers (above or below the box) are outliers.

#   - scatterplot - for bivariate outlier detection
plt.figure(figsize=(6,4))
sns.scatterplot(x='Age', y='Salary', data=df)
plt.title('Scatterplot: Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

# points that are far away from the cluster of other points are outliers.

#   - histogram - for detecting distributional skewness and extreme values
plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=10, kde=True)
plt.title('Histogram of Age')
plt.show()

# long tails or bars that are far from the peak suggest possible outliers.

# how to treat outliers

# drop outliers
df_no_outliers = df[(df['Salary'] >= lower_bound) & (df['Salary'] <= upper_bound)]

# clip data at extremes
lower = df['Salary'].quantile(0.05)
upper = df['Salary'].quantile(0.95)

df['Salary_capped'] = df['Salary'].clip(lower, upper)

#log transform (only for positive data)
import numpy as np
df['Log_Salary'] = np.log(df['Salary'])



#5. Feature Encoding
# feature encoding is the process of converting categorical data into numerical form so that machine learning models can process it
# types
# -> LABEL ENCODING : Ordinal data (with order)
# -> ONE-HOT ENCODING : Nominal data (no order)
# -> ORDINAL ENCODING : Manual ranking

# Label Encoding
# - good for ordinal categories (e.g., Low < Medium < High)
# - dont use label encoding on nominal data with no order, may mislead models into thinking Islamabad < Karachi < Lahore 

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Education_Encoded'] = le.fit_transform(df['Education'])


# One-Hot Encoding
# - good for nominal categories (no order)

df_onehot = pd.get_dummies(df, columns=['City'], drop_first=True)

"""    
    Name       City
0    Ali     Lahore
1   Sara    Karachi
2  Ahmed  Islamabad
3   Zain     Lahore

- Case 1: drop_first=False (default)

pd.get_dummies(df, columns=["City"], drop_first=False)

- Output:

    Name   City_Islamabad  City_Karachi  City_Lahore
0    Ali        0              0            1
1   Sara        0              1            0
2  Ahmed        1              0            0
3   Zain        0              0            1

- Here, 3 dummy columns are created (one for each city).
    - This is fine for tree models.
    - But for linear/logistic regression, these 3 columns are redundant (perfect multicollinearity).

- Case 2: drop_first=True

pd.get_dummies(df, columns=["City"], drop_first=True)

- Output:

    Name   City_Islamabad  City_Karachi
0    Ali        0              0            # Lahore (dropped category)
1   Sara        0              1            # Karachi
2  Ahmed        1              0            # Islamabad
3   Zain        0              0            # Lahore

- Now, only 2 dummy columns are created.
    - If both are 0 → Lahore (the dropped base category).
    - If City_Karachi=1 → Karachi.
    - If City_Islamabad=1 → Islamabad.
    
    -> This avoids redundancy in regression models.

-> drop_first=False: Creates dummy columns for all categories (good for trees).
-> drop_first=True: Drops one category column to avoid dummy variable trap (good for regression).

-> LabelEncoder: assigns IDs automatically (alphabetical order).
-> One-Hot Encoding: creates binary columns (good for nominal categories).
-> Ordinal Encoding (custom mapping): lets you define the order explicitly (good for ordered categories).

"""

# Ordinal Encoding (custom mapping)
education_map = {'Bachelors': 1, 'Masters': 2, 'PhD': 3}
df['Education_Ordinal'] = df['Education'].map(education_map)
print(df)

# Univariate, Bivariate, & Multivariate Analysis

# UNIVARIATE - Analysis of one variable at a time focuses on understanding distribution, central tendency, spread and outliers
# use cases:
#   - finding mean, median, mode
#   - detecting outliers (boxplots)
#   - understand shape (skewed, normal)

# summary statistics
print(df['Age'].describe())

# histogram
sns.histplot(df['Age'], bins=10, kde=True)
plt.title("Histogram of Age")
plt.show()

# boxplot
sns.boxplot(x=df['Age'])
plt.title("boxplot of age")
plt.show()

# BIVARIATE - analysis of two variables to examine the relationship between them
# use cases:
#   - correlation (numeric-numeric)
#   - group differences (numeric-categorical)
#   - scatter patterns (linear/nonlinear)

# correlation
print("correlation: ", df.corr(numeric_only=True))

# scatterplot
sns.scatterplot(x='Age', y='Salary', data=df)
plt.title("age vs salary")
plt.show()

# MULTIVARIATE - analysis involving 3 or more variables simulataneously to understand complex interactions
# use cases:
#   - heatmaps
#   - pairplots
#   - multiple regression
#   - PCA (dimensionality reduction)

# Handling Imbalanced Data
#   - class imbalance occurs when the target variable in a classification task has unequal class distribution
#   - in such a case, the model learns to predict only the majority class to maximize its accuracy
#       - e.g. 95% of labels are class 0 (NORMAL), & only 5% are class 1 (FRAUD)

# 1. OverSampling

from imblearn.over_sampling import RandomOverSampler

#perform random oversampling
ros = RandomOverSampler(random_state=0)
X_train_ros, y_train_ros = ros.fit_resample(X_train_pca, y_train)

# 2. UnderSampling

from imblearn.under_sampling import RandomUnderSampler

#perform random sampling
ros = RandomUnderSampler(random_state=0)
X_train_rus, y_train_rus = rus.fit_resample(X_train_pca, y_train)

"""
⚠ Problem: Imbalanced Data

- When one class (e.g., Normal = 95%) dominates the dataset compared to another (e.g., Fraud = 5%).
- A model trained on this imbalance might ignore the minority class and always predict the majority, giving misleadingly high accuracy but poor fraud detection.

✅ Solution 1: OverSampling

- RandomOverSampler duplicates minority class examples until classes are balanced.
- After this, the model has enough fraud cases to learn patterns.

✅ Solution 2: UnderSampling

RandomUnderSampler reduces the majority class by randomly removing some examples.
This balances the dataset but at the cost of losing information from the majority class.

in conclusion:
-> Oversampling = add more minority samples (risk: overfitting).
-> Undersampling = remove majority samples (risk: losing data).


"""
