# Machine Learning and Big Data - (Tree-Based Modeling)

This repository contains the complete code, analysis, and write-up for **Tree-Based Modeling**  group project in my **Machine Learning and Big Data** course at United States International University-Africa (USIU-Africa). The project explores and applies tree-based machine learning algorithms (Random Forest, Gradient Boosting, and Decision Trees) to real world classification and regression problems using Python and scikit-learn.



## Project Overview & Datasets

The project is structured into two main tasks, following a rigorous end-to-end data science pipeline (loading, exploratory data analysis, cleaning, preprocessing, baseline modeling, hyperparameter tuning, evaluation, and critical discussion):

### Part A - Task 1: COVID-19 Risk Classification
* **Objective:** Predict whether an individual is likely to test positive for COVID-19 (`covid19_test_results`) based on demographics, comorbidity flags, and symptoms[cite: 59].
* **Dataset:** `coronavirusdataset.csv` (7,294 rows, 45 initial columns)[cite: 59, 60].
* **Models Evaluated:** Dummy Classifier (majority baseline), Random Forest Classifier, and Gradient Boosting Classifier[cite: 67].
* **Key Challenges & Solutions:** 
  * Handled extreme class imbalance (~0.34% positive rate, with only 25 positive cases total) using `class_weight="balanced"`, custom sample weighting, and probability threshold tuning optimized on training F1 scores[cite: 62, 67, 69].
  * Dropped sparse columns (>70% to 99% missing data such as chest X-ray fields, rapid flu/strep tests, and missing vitals) to avoid fabricating clinical data[cite: 61, 63].
  * Evaluated performance using Precision, Recall, F1-score on the positive class, confusion matrices, and 5-fold stratified cross-validated ROC-AUC rather than misleading overall accuracy[cite: 62, 67, 70].

### Part B - Task 2: Car Price Prediction (Regression)
* **Objective:** Predict used automobile market prices based on vehicle attributes, registration year, mileage (`odometer`), and power specifications (`powerPS`)[cite: 72, 75].
* **Dataset:** `Auto Dataset.csv` (50,000 raw German eBay-style used-car listings)[cite: 73, 74].
* **Models Evaluated:** Dummy Regressor (mean baseline), Unpruned Decision Tree Regressor, and Tuned Decision Tree Regressor (`max_depth`, `min_samples_leaf`, `min_samples_split`)[cite: 77, 78].
* **Key Challenges & Solutions:**
  * Cleaned price and odometer strings by stripping currency symbols and units[cite: 74, 75].
  * Filtered out unrealistic market listings, placeholder zeros, and extreme price outliers (retaining realistic used cars priced €500–€40,000, registration years 1990–2016, and horsepower between 40 and 400 PS, totaling 38,313 clean rows)[cite: 76].
  * Evaluated models using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and coefficient of determination ($R^2$), and visualized tree structures (`max_depth=4` and tuned depth 12) to interpret market pricing drivers[cite: 77, 78, 79].



## Technical Stack & Workflow

* **Language:** Python 3.x[cite: 58]
* **Libraries:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`[cite: 58]
* **Core Methodologies:**
  * **Scale-Invariance:** Leveraging tree-based models' native robustness to feature scales without requiring standardization[cite: 66].
  * **Missing Value Handling:** Using `SimpleImputer(strategy="most_frequent")` for remaining categorical flags while dropping columns with excessive missingness[cite: 63, 66].
  * **Hyperparameter Optimization:** Employing `GridSearchCV` with Stratified K-Fold cross-validation[cite: 68, 69, 77].
  * **Model Interpretation:** Extracting feature importances, plotting confusion matrices, analyzing residual distributions, and exporting text-based and graphical decision tree rules[cite: 70, 71, 78, 79, 80].


## Running the code
This analysis is contained within the Jupyter Notebook. You can view it directly on GitHub or run it locally by launching Jupyter.
