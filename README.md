# Heart Disease AI Predictor

A complete end-to-end Machine Learning system to predict heart disease using clinical patient data.

---

## Project Structure

```
Heart_Disease_AI_System/
│
├── Dataset/
│       values.csv          ← Patient clinical features (180 rows × 14 cols)
│       labels.csv          ← Target labels (heart_disease_present: 0/1)
│
├── Saved_Model/
│       best_model.pkl      ← Trained best model (Random Forest)
│       scaler.pkl          ← StandardScaler fitted on training data
│       encoder.pkl         ← LabelEncoder fitted on thal column
│
├── Reports/
│       prediction_report.pdf  ← Auto-generated PDF analysis report
│
├── Images/
│       charts/             ← All EDA and model evaluation charts saved as PNG
│
├── Notebook/
│       Heart_Disease_AI_System.ipynb  ← Full Jupyter notebook (all tasks)
│
├── app.py                  ← Flask web application for live predictions
├── requirements.txt        ← All Python dependencies
└── README.md               ← This file
```

---

## How to Run

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the Jupyter Notebook
```bash
cd Notebook
jupyter notebook Heart_Disease_AI_System.ipynb
```
Run all cells top to bottom. This will:
- Perform full EDA and save charts to `Images/charts/`
- Train 6 ML models and compare them
- Save the best model to `Saved_Model/`
- Generate `Reports/prediction_report.pdf`

### Step 3 — Launch the Web App
```bash
python app.py
```
Open your browser at: **http://127.0.0.1:5000**

Enter patient details in the form and get an instant heart disease risk prediction.

---

## Dataset Description

| Column | Type | Description |
|--------|------|-------------|
| slope_of_peak_exercise_st_segment | int | Slope of peak exercise ST segment |
| thal | categorical | Thallium stress test result (normal / fixed_defect / reversible_defect) |
| resting_blood_pressure | int | Resting blood pressure (mmHg) |
| chest_pain_type | int | Chest pain type (1-4) |
| num_major_vessels | int | Number of major vessels (0-3) colored by fluoroscopy |
| fasting_blood_sugar_gt_120_mg_per_dl | binary | Fasting blood sugar > 120 mg/dl |
| resting_ekg_results | int | Resting EKG results (0, 1, 2) |
| serum_cholesterol_mg_per_dl | int | Serum cholesterol (mg/dl) |
| oldpeak_eq_st_depression | float | ST depression induced by exercise |
| sex | binary | 0 = Female, 1 = Male |
| age | int | Age in years |
| max_heart_rate_achieved | int | Maximum heart rate achieved (bpm) |
| exercise_induced_angina | binary | Exercise-induced chest pain (0/1) |

**Target:** `heart_disease_present` → 0 = No Disease, 1 = Disease Present

---

## Model Performance

| Model | Accuracy | Recall | F1-Score | ROC-AUC |
|-------|----------|--------|----------|---------|
| Random Forest | 0.850 | 0.850 | 0.850 | **0.950** |
| XGBoost | 0.925 | 1.000 | 0.930 | 0.947 |
| SVM | 0.825 | 0.900 | 0.837 | 0.922 |
| KNN | 0.825 | 0.850 | 0.829 | 0.922 |
| Logistic Regression | 0.825 | 0.850 | 0.829 | 0.892 |
| Decision Tree | 0.750 | 0.800 | 0.762 | 0.714 |

**Best Model: Random Forest (ROC-AUC = 0.95)**

---

## Domain
Healthcare — Cardiovascular Disease Prediction

