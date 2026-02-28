# Titanic Survival Prediction - End-to-End ML Project

## 📊 Project Overview

This project builds a complete machine learning pipeline to predict Titanic passenger survival using Python, Pandas, and Scikit-learn. It demonstrates best practices in data science workflow: exploratory data analysis (EDA), feature engineering, model comparison, and evaluation.

**Target Accuracy:** 82-84%  
**Timeline:** 7-10 days  
**Difficulty Level:** Beginner-Intermediate

---

## 📁 Project Structure

```
titanic-survival-prediction/
├── notebooks/
│   ├── EDA.ipynb                    # Exploratory Data Analysis
│   └── model_training.ipynb         # Feature engineering & model training
├── data/
│   ├── train.csv                    # Training dataset
│   └── test.csv                     # Test dataset
├── plots/
│   ├── 01_survival_distribution.png
│   ├── 02_survival_by_pclass.png
│   ├── 03_survival_by_sex.png
│   ├── 04_survival_by_age.png
│   ├── 05_survival_by_fare.png
│   ├── 06_family_analysis.png
│   ├── 07_embarked_analysis.png
│   ├── 08_correlation_heatmap.png
│   ├── 09_model_comparison.png
│   ├── 10_confusion_matrices.png
│   ├── 11_roc_curves.png
│   └── 12_feature_importance.png
├── results/
│   └── submission.csv               # Final predictions
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone/create project folder
mkdir titanic-survival-prediction
cd titanic-survival-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

Download Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic):
- `train.csv` (891 passengers)
- `test.csv` (418 passengers)

Place files in `data/` folder.

### 3. Run Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/EDA.ipynb first
# Then notebooks/model_training.ipynb
```

---

## 📚 Notebook Breakdown

### **Notebook 1: EDA.ipynb** (2-3 days)

**Goals:**
- Understand data structure and distributions
- Identify missing values and patterns
- Discover feature relationships with survival

**Key Sections:**
1. **Data Loading & Inspection** - Shape, types, summary statistics
2. **Missing Value Analysis** - Identify gaps in data
3. **Survival Distribution** - 38% survived, 62% didn't
4. **Demographic Analysis:**
   - **Sex**: 74% females survived vs 19% males (strongest predictor)
   - **Class**: 1st class 63% survival vs 3rd class 24%
   - **Age**: Children more likely to survive
   - **Fare**: Higher fares correlate with survival
5. **Family Relations** - Solo travelers less likely to survive
6. **Embarked Port** - Port of embarkation affects survival
7. **Correlation Heatmap** - Feature relationships

**Key Insights:**
```
✓ Women and children prioritized (74% female survival)
✓ 1st class passengers had better access to lifeboats
✓ Younger passengers more likely to survive
✓ Higher fares = better survival (proxy for class)
✓ Solo travelers had lower survival rates
```

**Visualizations:** 12+ plots including heatmaps, distributions, box plots

---

### **Notebook 2: model_training.ipynb** (4-7 days)

**Goals:**
- Prepare data for modeling
- Build and compare multiple models
- Evaluate performance with multiple metrics
- Generate predictions

**Key Sections:**

#### 1. Feature Engineering (Most Important!)
```python
✓ Age imputation by Pclass & Sex (more accurate than mean)
✓ Categorical encoding (Sex: 0/1, Embarked: 0/1/2)
✓ FamilySize = SibSp + Parch + 1
✓ IsAlone = (FamilySize == 1)
✓ FarePerPerson = Fare / FamilySize
✓ AgeGroup = Binned into 5 categories
✓ FareBin = Quartile binning
✓ Title extraction from Name (Mr, Mrs, Master, etc.)
✓ IsChild = (Age < 18)
```

#### 2. Data Preparation
- Train/test split: 80/20 with stratification (preserves class ratio)
- Feature scaling for Logistic Regression (StandardScaler)
- No scaling needed for tree-based models

#### 3. Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** ✓ | **0.8371** | **0.8205** | **0.7500** | **0.7843** | **0.8949** |
| Gradient Boosting | 0.8212 | 0.8167 | 0.6875 | 0.7467 | 0.8790 |
| Logistic Regression | 0.8044 | 0.7976 | 0.6375 | 0.7099 | 0.8709 |

**Winner: Random Forest** 🏆

#### 4. Evaluation Metrics

- **Accuracy**: Overall correctness (84%)
- **Precision**: Of predicted survivors, 82% actually survived
- **Recall**: Of actual survivors, 75% were correctly identified
- **F1-Score**: Harmonic mean balancing precision & recall
- **ROC-AUC**: 0.89 (excellent discrimination)

#### 5. Feature Importance (Top Features)

1. **Sex** (23%) - Most important
2. **Fare** (18%)
3. **Age** (15%)
4. **Pclass** (12%)
5. **FamilySize** (8%)
6. **Title** (7%)

#### 6. Visualizations Generated

- Confusion matrices (all 3 models)
- ROC curves (model comparison)
- Feature importance bar chart
- Model performance comparison

---

## 🔧 Implementation Steps (Day-by-Day)

### **Days 1-2: EDA**
- [ ] Load data and inspect
- [ ] Analyze missing values
- [ ] Create survival distribution plots
- [ ] Explore relationships with key features
- [ ] Generate correlation heatmap
- [ ] Document insights

### **Days 3-4: Feature Engineering**
- [ ] Handle missing values (Age, Embarked, Fare)
- [ ] Encode categorical variables
- [ ] Create new features (FamilySize, IsAlone, AgeGroup)
- [ ] Feature scaling for linear models
- [ ] Validate no data leakage

### **Days 5-6: Model Training**
- [ ] Train 3 models (Logistic Regression, Random Forest, Gradient Boosting)
- [ ] Perform cross-validation
- [ ] Generate predictions
- [ ] Create confusion matrices and ROC curves

### **Days 7-8: Evaluation & Optimization**
- [ ] Compare model metrics
- [ ] Extract feature importance
- [ ] Analyze misclassifications
- [ ] Document results

### **Days 9-10: Finalization**
- [ ] Create submission file
- [ ] Write comprehensive README
- [ ] Document code with comments
- [ ] Push to GitHub

---

## 📊 Expected Results

```
✓ Train Accuracy: ~84%
✓ Test Accuracy: ~80-82% (on public Kaggle leaderboard)
✓ ROC-AUC: ~0.89 (excellent)
✓ Kaggle Submission Rank: Top 20-30% of submissions
```

---

## 💡 Key Skills Demonstrated

### Python & Data Science
- ✅ Data loading and inspection with Pandas
- ✅ Exploratory data analysis (EDA)
- ✅ Missing value imputation strategies
- ✅ Feature engineering and transformation
- ✅ Train/test splitting and stratification

### Machine Learning
- ✅ Logistic Regression
- ✅ Random Forest Classification
- ✅ Gradient Boosting
- ✅ Cross-validation
- ✅ Hyperparameter tuning basics

### Evaluation & Metrics
- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ Confusion matrices
- ✅ ROC curves and AUC
- ✅ Feature importance analysis

### Visualization
- ✅ Histograms and distributions
- ✅ Box plots and violin plots
- ✅ Correlation heatmaps
- ✅ ROC curves
- ✅ Feature importance charts

### Professional Practices
- ✅ Code organization and structure
- ✅ Documentation with markdown
- ✅ GitHub repository setup
- ✅ README documentation

---

## 🎯 Resume Bullet Points

Add these to your resume:

```
✓ Titanic Survival Prediction - End-to-End ML Pipeline
  - Engineered 12+ features from raw data (family size, age groups, 
    fare per person, title extraction)
  - Built and compared 3 classification models: Logistic Regression, 
    Random Forest, Gradient Boosting
  - Achieved 83.7% accuracy using Random Forest with ROC-AUC of 0.895
  - Performed comprehensive EDA with 12+ visualizations identifying 
    key survival predictors (sex, class, age, fare)
  - Implemented proper train/test split with stratification and 
    5-fold cross-validation
  - Skills: Python, Pandas, Scikit-learn, Matplotlib, Seaborn, 
    Feature Engineering, Classification Models
```

---

## 📚 Advanced Topics (Optional)

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```

### Ensemble Methods
```python
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(
    estimators=[('lr', lr_model), ('rf', rf_model), ('gb', gb_model)],
    voting='soft'
)
voting.fit(X_train, y_train)
```

### Class Imbalance Handling
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', 
                                    np.unique(y_train), y_train)
```

---

## 🔗 Resources

- **Kaggle Competition**: https://www.kaggle.com/c/titanic
- **Dataset Description**: Feature definitions and info
- **Scikit-learn Docs**: https://scikit-learn.org
- **Pandas Tutorial**: https://pandas.pydata.org/docs
- **Matplotlib Guide**: https://matplotlib.org

---

## 📈 Performance Benchmark

| Milestone | Target | Status |
|-----------|--------|--------|
| EDA Completion | 2-3 days | ✓ |
| Feature Engineering | 3-4 days | ✓ |
| Model Training | 1-2 days | ✓ |
| Evaluation & Tuning | 1 day | ✓ |
| **Total Duration** | **7-10 days** | ✓ |

---

## 🎓 Learning Outcomes

After completing this project, you'll understand:

1. **Complete ML Workflow** - From raw data to predictions
2. **EDA Best Practices** - How to explore and understand data
3. **Feature Engineering** - Creating meaningful features from raw data
4. **Model Selection** - When and why to use different algorithms
5. **Evaluation Metrics** - How to properly assess model performance
6. **Data Preprocessing** - Handling missing values and encoding
7. **Cross-Validation** - Avoiding overfitting and ensuring generalization
8. **Visualization** - Communicating results effectively

---

## 🤝 Contributing

Found a bug or improvement? Feel free to submit a pull request!

---

## 📄 License

This project is open source and available under MIT License.

---

## 👤 Author

Created as a foundational ML project for data science learning by Prathamesh Bhamare.

**Last Updated:** November 2025
