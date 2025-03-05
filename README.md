# 🏠 House Prices - Advanced Regression Techniques

## 📌 Project Overview
This project is part of the Kaggle competition **"House Prices - Advanced Regression Techniques"**, where the goal is to predict the selling price of houses based on multiple features.

We follow a **professional machine learning pipeline** that includes **data preprocessing, outlier detection, feature encoding, correlation analysis, and model training.**

---

## 📂 Dataset Information
### 🔹 **Train Dataset** (`train.csv`)
- Contains **1460 observations** and **80 features**.
- Includes **SalePrice** as the target variable.

### 🔹 **Test Dataset** (`test.csv`)
- Contains **1459 observations** and **79 features**.
- Missing **SalePrice**, which needs to be predicted.

---

## 🔍 Data Preprocessing Steps

### ✅ **Handling Missing Values**
**1. Categorical Features:**
- `PoolQC`, `Alley`, `Fence`, `FireplaceQu`, `MiscFeature` → Filled with **"None"** (Indicating absence)
- `Electrical` → Filled with **mode** (most frequent value)
- `GarageQual`, `GarageType`, `GarageCond`, `GarageFinish` → Filled with **"None"**
- `KitchenQual`, `MSZoning`, `SaleType`, `Utilities` → Filled with **mode**
- `MasVnrType` → Filled with **"None"**

**2. Numerical Features:**
- `GarageYrBlt`, `MasVnrArea` → Filled with **median**
- `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`, `BsmtFullBath`, `BsmtHalfBath` → Filled with **0** (assuming no basement)
- `GarageCars`, `GarageArea` → Filled with **0** (assuming no garage)

**3. `LotFrontage` Feature Imputation:**
- **Used `RandomForestRegressor`** to predict missing values based on `LotArea` and `Neighborhood`.

### ✅ **Final Check for Missing Values**
- Ensured that **all missing values** were filled using `df.isnull().sum().sum() == 0`.

---

## 📊 Exploratory Data Analysis

### ✅ **Correlation Analysis**
- **Selected only numerical features**: `df.select_dtypes(include=[np.number])`
- Computed the **correlation matrix**: `df_train_numeric.corr()`
- **Top 10 features most correlated with `SalePrice`** were identified and visualized using `seaborn` bar plot.

### ✅ **Outlier Detection**
1. **Boxplot Visualization** for `SalePrice`
   - Used `sns.boxplot()` to detect extreme price variations.
2. **Z-Score Method** for Outlier Detection
   - Identified outliers in `SalePrice`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `1stFlrSF`.
   - Defined **outliers as values with `Z-score > 3`**.
   - Counted outliers for each feature.

---

## 🔄 Feature Encoding

### ✅ **Label Encoding & One-Hot Encoding**
1. **Label Encoding (For categorical features with ≤ 3 unique values)**
   - Used `LabelEncoder()` to transform categorical values into numeric ones.
   - Handled new categories in `test.csv` by replacing them with **"Unknown"**.

2. **One-Hot Encoding (For categorical features with more than 3 unique values)**
   - Applied `pd.get_dummies()` to create binary features for each category.
   - Ensured **`train` and `test` had the same columns** by adding missing ones with **0 values**.

3. **Final check to ensure no categorical features remain**
   - Used `df.select_dtypes(include=["object"]).columns` to confirm successful encoding.

---

## 🚀 Model Training (Next Steps)

Now that the dataset is **fully cleaned and processed**, we are ready to:
1. **Feature Engineering** - Scale/Transform features if needed.
2. **Model Selection** - Train multiple regression models like `XGBoost`, `RandomForest`, `Linear Regression`.
3. **Hyperparameter Tuning** - Use `GridSearchCV` or `RandomizedSearchCV` to optimize model performance.
4. **Prediction & Submission** - Make final predictions and submit results to Kaggle.

---

## 📌 Final Thoughts
- This project followed a **structured data processing pipeline** ensuring high-quality input for machine learning models.
- **All missing values handled, outliers identified, and categorical features encoded properly.**
- **Next steps include model training and evaluation to maximize prediction accuracy.** 🚀🔥


## 🚀 Why Participate?  
✔ **Great for beginners** - Ideal for getting started with machine learning  
✔ **Rolling leaderboard** - Compete continuously and improve your skills  
✔ **Exploratory data analysis** - Gain hands-on experience with real-world datasets  
✔ **Model optimization** - Experiment with different classification algorithms  

## 🏆 How to Get Started?  
1. Explore the dataset on Kaggle  
2. Perform data cleaning & feature engineering  
3. Train and evaluate machine learning models  
4. Submit your predictions & improve your ranking  

📌 **Official Kaggle Page:** [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
