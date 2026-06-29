"""
app.py — Heart Disease AI System
Flask web application for live heart disease risk prediction.

Run:
    python app.py
Then open: http://127.0.0.1:5000
"""

import os, joblib
import numpy as np
from flask import Flask, request, render_template_string

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
MODELS  = os.path.join(BASE, 'Saved_Model')

# ── Load saved artefacts ──────────────────────────────────────────────────────
model   = joblib.load(os.path.join(MODELS, 'best_model.pkl'))
scaler  = joblib.load(os.path.join(MODELS, 'scaler.pkl'))
encoder = joblib.load(os.path.join(MODELS, 'encoder.pkl'))

app = Flask(__name__)

# ── HTML Template (single-file, no external templates folder needed) ──────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Heart Disease AI System</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', Tahoma, sans-serif;
      background: linear-gradient(135deg, #fff5f5 0%, #ffe3e3 100%);
      min-height: 100vh; padding: 30px 15px;
    }
    .container { max-width: 780px; margin: 0 auto; }

    .header {
      background: linear-gradient(135deg, #c92a2a, #e03131);
      color: white; padding: 28px 32px; border-radius: 14px;
      margin-bottom: 24px; text-align: center;
      box-shadow: 0 4px 20px rgba(201,42,42,0.3);
    }
    .header h1 { font-size: 28px; margin-bottom: 6px; }
    .header p  { font-size: 14px; opacity: 0.88; }

    .card {
      background: white; border-radius: 14px; padding: 28px 32px;
      box-shadow: 0 2px 16px rgba(0,0,0,0.08); margin-bottom: 24px;
    }
    .card h2 { font-size: 16px; color: #c92a2a; margin-bottom: 18px;
               border-bottom: 2px solid #ffe3e3; padding-bottom: 8px; }

    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    label { display: block; font-size: 12px; color: #555; margin-bottom: 4px; font-weight: 600; }
    input, select {
      width: 100%; padding: 9px 12px; border: 1.5px solid #dee2e6;
      border-radius: 8px; font-size: 13px; transition: border .2s;
    }
    input:focus, select:focus { outline: none; border-color: #c92a2a; }

    .btn {
      width: 100%; padding: 14px; background: linear-gradient(135deg,#c92a2a,#e03131);
      color: white; border: none; border-radius: 10px; font-size: 16px;
      font-weight: 700; cursor: pointer; transition: transform .15s, box-shadow .15s;
      margin-top: 8px;
    }
    .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(201,42,42,0.35); }

    .result { border-radius: 14px; padding: 24px 28px; text-align: center; }
    .result.positive { background: #fff5f5; border: 2px solid #ff6b6b; }
    .result.negative { background: #f0fff4; border: 2px solid #51cf66; }
    .result .emoji  { font-size: 52px; margin-bottom: 8px; }
    .result .verdict { font-size: 22px; font-weight: 800; margin-bottom: 6px; }
    .result.positive .verdict { color: #c92a2a; }
    .result.negative .verdict { color: #2f9e44; }
    .result .prob { font-size: 15px; color: #555; margin-bottom: 12px; }
    .bar-wrap { background: #f1f3f5; border-radius: 50px; height: 14px;
                overflow: hidden; margin: 10px 0; }
    .bar-fill  { height: 100%; border-radius: 50px; transition: width 1s; }
    .bar-fill.high { background: linear-gradient(90deg,#ff8787,#c92a2a); }
    .bar-fill.low  { background: linear-gradient(90deg,#8ce99a,#2f9e44); }
    .note { font-size: 11px; color: #888; margin-top: 10px; }

    .footer { text-align: center; font-size: 12px; color: #aaa; margin-top: 10px; }
  </style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>&#10084;&#65039; Heart Disease AI System</h1>
    <p>PRCP-1016 | Enter patient clinical data to get an instant risk prediction</p>
  </div>

  <div class="card">
    <h2>Patient Clinical Data</h2>
    <form method="POST" action="/predict">
      <div class="grid">

        <div>
          <label>Age (years)</label>
          <input type="number" name="age" min="20" max="90"
                 value="{{ form.get('age','54') }}" required/>
        </div>
        <div>
          <label>Sex</label>
          <select name="sex">
            <option value="1" {% if form.get('sex')=='1' %}selected{% endif %}>Male</option>
            <option value="0" {% if form.get('sex')=='0' %}selected{% endif %}>Female</option>
          </select>
        </div>

        <div>
          <label>Chest Pain Type (1-4)</label>
          <select name="chest_pain_type">
            {% for v,l in [('1','1 – Typical Angina'),('2','2 – Atypical Angina'),
                           ('3','3 – Non-Anginal'),('4','4 – Asymptomatic')] %}
            <option value="{{v}}" {% if form.get('chest_pain_type','')==v %}selected{% endif %}>{{l}}</option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label>Resting Blood Pressure (mmHg)</label>
          <input type="number" name="resting_blood_pressure" min="80" max="220"
                 value="{{ form.get('resting_blood_pressure','130') }}" required/>
        </div>

        <div>
          <label>Serum Cholesterol (mg/dl)</label>
          <input type="number" name="serum_cholesterol_mg_per_dl" min="100" max="600"
                 value="{{ form.get('serum_cholesterol_mg_per_dl','245') }}" required/>
        </div>
        <div>
          <label>Fasting Blood Sugar &gt; 120 mg/dl</label>
          <select name="fasting_blood_sugar_gt_120_mg_per_dl">
            <option value="0" {% if form.get('fasting_blood_sugar_gt_120_mg_per_dl','0')=='0' %}selected{% endif %}>No (&le;120)</option>
            <option value="1" {% if form.get('fasting_blood_sugar_gt_120_mg_per_dl','0')=='1' %}selected{% endif %}>Yes (&gt;120)</option>
          </select>
        </div>

        <div>
          <label>Resting EKG Results</label>
          <select name="resting_ekg_results">
            {% for v,l in [('0','0 – Normal'),('1','1 – ST-T Abnormality'),('2','2 – LV Hypertrophy')] %}
            <option value="{{v}}" {% if form.get('resting_ekg_results','')==v %}selected{% endif %}>{{l}}</option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label>Max Heart Rate Achieved (bpm)</label>
          <input type="number" name="max_heart_rate_achieved" min="60" max="220"
                 value="{{ form.get('max_heart_rate_achieved','150') }}" required/>
        </div>

        <div>
          <label>Exercise Induced Angina</label>
          <select name="exercise_induced_angina">
            <option value="0" {% if form.get('exercise_induced_angina','0')=='0' %}selected{% endif %}>No</option>
            <option value="1" {% if form.get('exercise_induced_angina','0')=='1' %}selected{% endif %}>Yes</option>
          </select>
        </div>
        <div>
          <label>ST Depression (oldpeak)</label>
          <input type="number" name="oldpeak_eq_st_depression" step="0.1" min="0" max="10"
                 value="{{ form.get('oldpeak_eq_st_depression','1.0') }}" required/>
        </div>

        <div>
          <label>Slope of Peak Exercise ST</label>
          <select name="slope_of_peak_exercise_st_segment">
            {% for v,l in [('1','1 – Upsloping'),('2','2 – Flat'),('3','3 – Downsloping')] %}
            <option value="{{v}}" {% if form.get('slope_of_peak_exercise_st_segment','')==v %}selected{% endif %}>{{l}}</option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label>Number of Major Vessels (0-3)</label>
          <select name="num_major_vessels">
            {% for v in ['0','1','2','3'] %}
            <option value="{{v}}" {% if form.get('num_major_vessels','')==v %}selected{% endif %}>{{v}}</option>
            {% endfor %}
          </select>
        </div>

        <div style="grid-column: span 2;">
          <label>Thalium Stress Test</label>
          <select name="thal">
            {% for v in ['normal','fixed_defect','reversible_defect'] %}
            <option value="{{v}}" {% if form.get('thal','')==v %}selected{% endif %}>{{v.replace('_',' ').title()}}</option>
            {% endfor %}
          </select>
        </div>

      </div><!-- /grid -->
      <button type="submit" class="btn">&#128200; Predict Heart Disease Risk</button>
    </form>
  </div><!-- /card -->

  {% if result is not none %}
  <div class="result {{ 'positive' if result==1 else 'negative' }}">
    <div class="emoji">{{ '🚨' if result==1 else '✅' }}</div>
    <div class="verdict">
      {{ 'Heart Disease DETECTED' if result==1 else 'No Heart Disease Detected' }}
    </div>
    <div class="prob">
      Predicted probability of disease: <strong>{{ prob }}%</strong>
    </div>
    <div class="bar-wrap">
      <div class="bar-fill {{ 'high' if result==1 else 'low' }}"
           style="width:{{ prob }}%"></div>
    </div>
    <p class="note">
      &#9888; This prediction is generated by an AI model for screening purposes only.
      Always consult a qualified medical professional for diagnosis and treatment.
    </p>
  </div>
  {% endif %}

  <div class="footer">Heart Disease AI System · PRCP-1016 · Model: {{ model_name }}</div>
</div>
</body>
</html>
"""

# ── Feature order must match training data ────────────────────────────────────
FEATURE_ORDER = [
    'slope_of_peak_exercise_st_segment',
    'thal',
    'resting_blood_pressure',
    'chest_pain_type',
    'num_major_vessels',
    'fasting_blood_sugar_gt_120_mg_per_dl',
    'resting_ekg_results',
    'serum_cholesterol_mg_per_dl',
    'oldpeak_eq_st_depression',
    'sex',
    'age',
    'max_heart_rate_achieved',
    'exercise_induced_angina'
]

MODEL_NAME = type(model).__name__


@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML, result=None, prob=None,
                                  form={}, model_name=MODEL_NAME)


@app.route('/predict', methods=['POST'])
def predict():
    form = request.form.to_dict()
    try:
        # Build feature vector in training order
        row = []
        for feat in FEATURE_ORDER:
            val = form.get(feat, '0')
            if feat == 'thal':
                # Encode using saved LabelEncoder
                val_enc = encoder.transform([val])[0]
                row.append(float(val_enc))
            else:
                row.append(float(val))

        X = np.array(row).reshape(1, -1)

        # Scale (scaler was fit on all features including thal)
        X_sc = scaler.transform(X)

        # Use scaled or raw depending on model type
        tree_models = ('RandomForestClassifier','XGBClassifier',
                       'DecisionTreeClassifier','GradientBoostingClassifier')
        if MODEL_NAME in tree_models:
            pred  = model.predict(X)[0]
            prob  = round(model.predict_proba(X)[0][1] * 100, 1)
        else:
            pred  = model.predict(X_sc)[0]
            prob  = round(model.predict_proba(X_sc)[0][1] * 100, 1)

        return render_template_string(HTML, result=int(pred), prob=prob,
                                      form=form, model_name=MODEL_NAME)

    except Exception as e:
        return render_template_string(HTML, result=None, prob=None,
                                      form=form, model_name=f'Error: {e}')


if __name__ == '__main__':
    print('='*55)
    print('  Heart Disease AI System — Flask App')
    print(f'  Model loaded : {MODEL_NAME}')
    print('  Open browser : http://127.0.0.1:5000')
    print('='*55)
    app.run(debug=True, port=5000)
