# Titanic ML Project - Quick Reference Guide

## 🎯 Project Goals
- Build end-to-end ML pipeline ✓
- Learn data preprocessing & feature engineering ✓
- Compare multiple ML models ✓
- Achieve 82-84% accuracy ✓
- Generate professional project deliverables ✓

---

## 📋 Step-by-Step Execution Plan

### WEEK 1: Foundation

#### Days 1-2: EDA Phase
**Notebook:** `notebooks/EDA.ipynb`

```python
# Key tasks:
✓ Load and inspect train.csv (891 rows, 11 features)
✓ Analyze missing values:
  - Age: 177 missing (19.9%)
  - Embarked: 2 missing (0.2%)
  - Cabin: 687 missing (77%) - DROP THIS
  
✓ Survival rate: 38% survived, 62% didn't

✓ Key patterns to discover:
  1. Sex: Females 74% survival vs Males 19%
  2. Class: 1st class 63%, 2nd class 47%, 3rd class 24%
  3. Age: Children (0-12) had high survival rate
  4. Fare: Higher fare = higher survival
  5. Family: Solo travelers died more
  
✓ Generate 12+ visualizations with seaborn
✓ Create correlation heatmap
✓ Document insights
```

**Expected Output:** 12 visualization files in `plots/`

---

#### Days 3-4: Feature Engineering
**Notebook:** `notebooks/model_training.ipynb` (First Section)

```python
# Feature Engineering Steps:

1. MISSING VALUE HANDLING
   - Age: Fill with median by Pclass & Sex (better than simple mean)
   - Embarked: Fill with mode ('S')
   - Cabin: Drop entirely (77% missing, too sparse)

2. CATEGORICAL ENCODING
   - Sex: male=0, female=1 (one-hot or direct mapping)
   - Embarked: S=0, C=1, Q=2
   - Title: Extract from Name (Mr=0, Miss=1, Mrs=2, Master=3, etc.)

3. NEW FEATURES (Feature Engineering)
   - FamilySize = SibSp + Parch + 1
   - IsAlone = (FamilySize == 1)
   - FarePerPerson = Fare / FamilySize
   - AgeGroup = Binned [0-12, 12-18, 18-35, 35-50, 50+]
   - FareBin = Quartiles
   - IsChild = (Age < 18)

4. DATA CLEANUP
   - Drop: PassengerId, Name, Ticket, Cabin
   - Fill any remaining NaNs with median

# Final feature set: ~18 features
```

**Expected Output:** Clean dataset ready for modeling

---

### Days 5-6: Model Building & Training
**Notebook:** `notebooks/model_training.ipynb` (Middle Sections)

```python
# Step 1: Prepare Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = train_fe.drop('Survived', axis=1)
y = train_fe['Survived']

# Train/Test Split: 80-20 with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 2: Scale Features (for linear models only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train 3 Models

## MODEL 1: Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
# Expected Accuracy: ~80%

## MODEL 2: Random Forest (BEST)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
# Expected Accuracy: ~83-84%

## MODEL 3: Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
# Expected Accuracy: ~82%

# Step 4: Cross-Validation (5-fold)
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"CV Score: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
```

**Expected Output:** Trained models with ~83% accuracy

---

### Days 7-8: Evaluation & Analysis
**Notebook:** `notebooks/model_training.ipynb` (Evaluation Sections)

```python
# Evaluation Metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_curve, 
                            roc_auc_score, classification_report)

# Comprehensive evaluation
accuracy = accuracy_score(y_test, rf_pred)      # ~83.7%
precision = precision_score(y_test, rf_pred)    # ~82%
recall = recall_score(y_test, rf_pred)          # ~75%
f1 = f1_score(y_test, rf_pred)                  # ~78%
auc = roc_auc_score(y_test, rf_pred_proba)      # ~0.89

# Visualizations
1. Confusion Matrix (3 subplots - one per model)
2. ROC Curves (all 3 models on same plot)
3. Feature Importance (top 10 features)
4. Model Comparison (accuracy, precision, recall, F1, AUC)

# Classification Report
print(classification_report(y_test, rf_pred, 
                          target_names=['Not Survived', 'Survived']))
```

**Expected Output:** 4-5 visualization files + performance metrics

---

### Days 9-10: Finalization & Submission
**Notebook:** `notebooks/model_training.ipynb` (Final Sections)

```python
# Step 1: Make Predictions on Test Data
X_test_final = test_fe.copy()
test_predictions = rf.predict(X_test_final)

# Step 2: Create Submission File
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('../results/submission.csv', index=False)

# Step 3: Finalize README.md
# Copy template from README_template.md
# Customize with your project details
# Include:
- Project overview
- File structure
- Setup instructions
- Results achieved
- Resume bullet points
- Key insights

# Step 4: Push to GitHub
git add .
git commit -m "Initial Titanic ML pipeline"
git push origin main
```

**Expected Output:** Kaggle submission ready

---

## 🎓 Key Concepts Quick Reference

### Feature Engineering
```
Raw Data → Missing Value Handling → Encoding → Feature Creation → Scaling → Ready for ML
```

### Model Comparison Metrics
| Metric | What it measures | Good value |
|--------|------------------|------------|
| **Accuracy** | Overall correctness | >80% |
| **Precision** | Of predicted positives, how many correct | >80% |
| **Recall** | Of actual positives, how many found | >75% |
| **F1-Score** | Harmonic mean of precision & recall | >0.78 |
| **ROC-AUC** | Discrimination ability across thresholds | >0.85 |

### When to Use Each Model
- **Logistic Regression**: Fast, interpretable, baseline comparison
- **Random Forest**: Good accuracy, handles non-linearity, robust
- **Gradient Boosting**: Highest accuracy potential, slower training

---

## 🔍 Debugging Checklist

**Issue:** Model accuracy is low (~50%)
- ✓ Check if target variable is properly loaded
- ✓ Verify train/test split ratio
- ✓ Ensure categorical variables are encoded
- ✓ Check for data leakage (test data info in training)

**Issue:** Missing values error
- ✓ Verify all imputation methods cover all columns
- ✓ Check for NaN after transformations
- ✓ Use `df.isnull().sum()` to find remaining gaps

**Issue:** Runtime errors in model training
- ✓ Ensure feature columns match between train and test
- ✓ Check data types (should be numeric for most models)
- ✓ Verify X and y have same number of rows

**Issue:** Features have very different scales
- ✓ Use StandardScaler before linear models
- ✓ Tree-based models don't need scaling

---

## 📊 Expected Results Summary

```
┌─────────────────────────────────────────┐
│ PERFORMANCE BENCHMARKS                  │
├─────────────────────────────────────────┤
│ Logistic Regression                     │
│   Accuracy:  80.44% ✓                   │
│   AUC:       0.8709 ✓                   │
│                                         │
│ Random Forest ⭐ BEST                   │
│   Accuracy:  83.71% ✓✓                  │
│   AUC:       0.8949 ✓✓                  │
│                                         │
│ Gradient Boosting                       │
│   Accuracy:  82.12% ✓                   │
│   AUC:       0.8790 ✓                   │
│                                         │
│ Kaggle Submission Score: 78-82%         │
└─────────────────────────────────────────┘
```

---

## 💼 Resume Bullet Point Template

**Replace XX% with actual improvement metric:**

```
Built an end-to-end ML pipeline for Titanic survival prediction with:
• Engineered 12+ features including family relationships, age groups, 
  and fare categories using Pandas
• Trained 3 classification models (Logistic Regression, Random Forest, 
  Gradient Boosting) and selected best performer
• Achieved 83.7% accuracy on test set using Random Forest with ROC-AUC 
  of 0.895
• Conducted comprehensive EDA identifying key predictors: sex (74% 
  female survival), class (1st: 63%, 3rd: 24%), and age
• Implemented proper data preprocessing including stratified train-test 
  split and 5-fold cross-validation
• Created 12+ publication-ready visualizations (heatmaps, ROC curves, 
  confusion matrices)
• Skills: Python, Pandas, Scikit-learn, Feature Engineering, 
  Classification Models, Data Visualization
```

---

## 🔗 File References

| File | Purpose | Size |
|------|---------|------|
| `notebooks/EDA.ipynb` | Exploratory analysis (12 visualizations) | ~80 KB |
| `notebooks/model_training.ipynb` | Feature engineering & modeling | ~120 KB |
| `plots/*.png` | 12 visualization outputs | ~15 MB total |
| `results/submission.csv` | Kaggle submission file | ~20 KB |
| `README.md` | Project documentation | ~50 KB |

---

## ⏱️ Time Allocation

- EDA & Analysis: 2-3 days (30%)
- Feature Engineering: 2-3 days (30%)
- Model Building: 1-2 days (20%)
- Evaluation & Tuning: 1 day (10%)
- Documentation: 1 day (10%)

**Total: 7-10 days**

---

## 🚀 After Project Completion

1. **Push to GitHub** with professional README
2. **Create Kaggle Notebook** version (link to notebook)
3. **Write Blog Post** explaining your approach
4. **Add to Portfolio** with live demo or notebook viewer link
5. **Practice Presentation** for interviews (explain your approach)

---

Good luck! 🎉
