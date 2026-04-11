HEART_CSV

Heart Disease Prediction — Cardiovascular Risk Assessment
Logistic Regression model for early cardiovascular risk detection using clinical features
![Python](https://img.shields.io/badge/Python-3.8+-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-LogisticRegression-orange) ![Domain](https://img.shields.io/badge/Domain-Cardiology-red) ![Status](https://img.shields.io/badge/Status-Complete-green)

Overview
This project develops a Logistic Regression model to predict the presence of heart disease using structured clinical and physiological data. The system simulates a cardiovascular risk screening tool that could assist clinicians in the early identification of high-risk patients — before acute events occur.
> Cardiovascular disease is the leading cause of death globally.  
> Most risk factors are already present in routine clinical data — but go unanalyzed.  
> This model puts that data to work.


Clinical Motivation
Challenge	Relevance
Late-stage detection	Many patients present only after a cardiac event
Underutilized risk data	Cholesterol, ECG, and exercise test results often sit in records unscored
Mortality burden	CVD accounts for ~32% of all global deaths (WHO)
Interpretability need	Clinicians require explainable predictions — not black-box outputs


Dataset
Property	Value
Source	Cleveland Heart Disease Database (UCI Machine Learning Repository)
Samples	303 patients
Features	13 clinical variables
Target	`target`: 1 = Heart Disease, 0 = No Heart Disease
Missing values	None
Feature Overview
Feature	Clinical Meaning	Notes
`age`	Patient age in years	Risk increases with age
`sex`	Biological sex (1 = male, 0 = female)	Males have higher baseline CVD risk
`cp`	Chest pain type (0–3)	Type 1 (typical angina) strongest indicator
`trestbps`	Resting blood pressure (mm Hg)	Hypertension elevates risk
`chol`	Serum cholesterol (mg/dl)	Long-term arterial risk marker
`fbs`	Fasting blood sugar > 120 mg/dl	Diabetes-CVD overlap signal
`restecg`	Resting ECG results (0–2)	Electrical activity of the heart
`thalach`	Maximum heart rate achieved	Lower max HR signals reduced cardiac fitness
`exang`	Exercise-induced angina (1 = yes)	Stress-related ischemia indicator
`oldpeak`	ST depression induced by exercise	ECG marker of myocardial ischemia
`slope`	Slope of peak exercise ST segment	Upsloping = healthier response
`ca`	Number of major vessels colored by fluoroscopy	Coronary artery blockage indicator
`thal`	Thalassemia type (0–3)	Blood disorder affecting oxygen delivery


ML Pipeline
```
Raw Clinical Data (303 patients)
        ↓
Exploratory Data Analysis
        ↓
Feature Standardization (StandardScaler)
        ↓
Train / Test Split (stratified 80/20)
        ↓
Logistic Regression Training
        ↓
Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
        ↓
Risk Prediction: Heart Disease vs No Heart Disease
```


Model
Why Logistic Regression?
Interpretable — feature coefficients directly express clinical variable influence
Probabilistic output — produces a risk probability (0–1), not just a binary flag
Clinically trusted — logistic regression underlies many validated clinical scoring systems (e.g. Framingham Risk Score)
Efficient — fast inference suitable for real-time clinical decision support
Baseline before ensemble models (Random Forest, XGBoost)


Results
Metric	Score
Training accuracy	84.3%
Test accuracy	80.3%
> Cross-validation and ROC-AUC metrics are planned for the next iteration.
Model Inference Example
```python
input_data = (44, 1, 1, 120, 263, 0, 1, 173, 0, 0.0, 2, 0, 3)
# Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
# RestingECG, MaxHR, ExerciseAngina, Oldpeak, Slope, CA, Thal

prediction = model.predict(input_data_reshaped)
# Output: The person has heart disease
```
Key Predictors
Feature	Direction	Clinical Significance
`cp` (chest pain type)	Typical angina → high risk	Primary diagnostic indicator
`thalach` (max heart rate)	Lower → higher risk	Reduced cardiac reserve
`exang` (exercise angina)	Present → higher risk	Stress-induced ischemia
`oldpeak` (ST depression)	Higher → higher risk	Myocardial ischemia marker
`ca` (vessel count)	More vessels → higher risk	Coronary artery disease burden
`chol`	Higher → higher risk	Long-term lipid accumulation
`age`	Older → higher risk	Cumulative cardiovascular exposure
> These predictors closely mirror established clinical diagnostic criteria — validating the model's clinical face validity.


Limitations
Limitation	Impact
Convergence warning	Logistic regression needs feature scaling and higher `max_iter`
No feature scaling applied (v1)	Unscaled features reduce coefficient reliability
Small dataset (303 samples)	Overfitting risk — cross-validation essential
Binary output only	No risk tier or probability threshold displayed to user
No ROC-AUC reported yet	AUC is the standard metric for clinical classifiers
Single demographic source	Cleveland cohort — generalizability to diverse populations limited


Planned Improvements
[ ] Apply `StandardScaler` before training — resolves convergence warning
[ ] Increase `max_iter` to 1000+ for full convergence
[ ] Add ROC-AUC curve and confusion matrix to evaluation
[ ] Compare with Random Forest and XGBoost
[ ] Implement SHAP feature importance for clinical explainability
[ ] Apply stratified k-fold cross-validation (5-fold)
[ ] Output probability score alongside binary prediction
[ ] Evaluate fairness across age and sex subgroups


Future Direction
Build a Streamlit risk calculator — clinician enters patient values, model returns risk score and top contributing factors
Extend to multi-class severity tiers: low / moderate / high / critical risk
Combine with Parkinson's and diabetes models in a unified multi-condition screening dashboard
Validate against external datasets (Framingham Heart Study cohort, NHANES)
---
What This Project Demonstrates
Clinical feature-based binary classification with interpretable model choice
Understanding of cardiovascular risk factors and their diagnostic significance
Recognition of model limitations (convergence, scaling, small N)
End-to-end ML workflow translatable to a real clinical decision-support tool


Tech Stack
`Python` `scikit-learn` `pandas` `NumPy` `Matplotlib` `Seaborn`


Related Projects
Project	Description
Diabetes Prediction Model	SVM classifier for Type 2 diabetes risk using Pima dataset
ASD Detection Model	Behavioral screening ML for Autism Spectrum Disorder
Parkinson's Detection Model	Voice biomarker SVM classifier for Parkinson's Disease
Dementia Risk Classification	Multi-class risk model with ~91% accuracy
Medicaid Drug Spending Dashboard	Power BI forecasting for Medicaid cardiovascular & diabetes drugs


References
Detrano R et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. American Journal of Cardiology.
UCI Machine Learning Repository — Heart Disease Dataset
World Health Organization — Cardiovascular Diseases Fact Sheet


Author
Prince Kwarteng Amaning  
MS Data Science (in progress) — University of Michigan–Dearborn  
Background: Dentistry | Global Public Health | Healthcare Analytics
Building interpretable clinical AI tools for early disease detection — with a focus on conditions where delayed diagnosis carries the highest human cost.
---
License
MIT License — open for educational and research use.
