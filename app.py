#  Build the Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Load saved model and scaler ───────────────────────────────────
model  = joblib.load('ckd_model.pkl')
scaler = joblib.load('ckd_scaler.pkl')

# ── Column orders ─────────────────────────────────────────────────
numerical_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr',
                  'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']

training_col_order = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
    'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
    'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
]

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title='CKD Prediction App',
    page_icon='🩺',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #f0f4ff; }
    .stButton>button {
        background-color: #4e73df;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px;
        width: 100%;
    }
    .stButton>button:hover { background-color: #2e59d9; }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 10px;
    }
    .ckd-box {
        background-color: #ffe0e0;
        border: 2px solid #e74c3c;
        color: #c0392b;
    }
    .no-ckd-box {
        background-color: #e0ffe0;
        border: 2px solid #2ecc71;
        color: #27ae60;
    }
    .metric-card {
        background-color: #4e73df;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        text-align: center;
        color: white !important;
    }
    .metric-card h3 {
        color: #d0e0ff !important;
        font-size: 15px;
        margin-bottom: 5px;
    }
    .metric-card h2 {
        color: white !important;
        font-size: 24px;
        margin: 0;
    }
    </style>
""", unsafe_allow_html=True)

# ── Session state for patient history ────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Sidebar Navigation ────────────────────────────────────────────
st.sidebar.image(
    "Kidney.jpg",
    width=250,
    caption="Empower your kidney health. Get fast, data-driven CKD risk insights in seconds"
)
st.sidebar.title('🩺 CKD Prediction App')
st.sidebar.markdown('---')

page = st.sidebar.radio(
    'Navigate',
    ['Home & Predict', '📋 Patient History', '📖 About & How to Use']
)

st.sidebar.markdown('---')
st.sidebar.markdown('### 📊 Model Info')
st.sidebar.info(
    '**Algorithm:** Random Forest\n\n'
    '**Accuracy:** 100%\n\n'
    '**Dataset:** UCI CKD Dataset\n\n'
    '**Features:** 24 clinical inputs'
)
st.sidebar.markdown('---')
st.sidebar.markdown(
    '<small>⚠️ This app is for research purposes only. '
    'Always consult a qualified physician.</small>',
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — HOME & PREDICT
# ══════════════════════════════════════════════════════════════════
if page == 'Home & Predict':

    st.title('🩺 Chronic Kidney Disease Prediction')
    st.markdown('Fill in the patient details below and click **Predict** to get a CKD risk assessment.')
    st.divider()

    # ── Input form ────────────────────────────────────────────────
    st.subheader('📋 Patient Clinical Details')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('**🔢 Numerical Features (Part 1)**')
        age  = st.number_input('Age (years)',            min_value=1,    max_value=100,   value=45)
        bp   = st.number_input('Blood Pressure (mm/Hg)', min_value=50,   max_value=180,   value=80)
        sg   = st.selectbox('Specific Gravity',          [1.005, 1.010, 1.015, 1.020, 1.025])
        al   = st.selectbox('Albumin',                   [0, 1, 2, 3, 4, 5])
        su   = st.selectbox('Sugar',                     [0, 1, 2, 3, 4, 5])
        bgr  = st.number_input('Blood Glucose Random',   min_value=50,   max_value=500,   value=100)
        bu   = st.number_input('Blood Urea',             min_value=10,   max_value=400,   value=40)
        sc   = st.number_input('Serum Creatinine',       min_value=0.0,  max_value=50.0,  value=1.0,  step=0.1)

    with col2:
        st.markdown('**🔢 Numerical Features (Part 2)**')
        sod  = st.number_input('Sodium',                 min_value=100,  max_value=170,   value=135)
        pot  = st.number_input('Potassium',              min_value=2.0,  max_value=50.0,  value=4.5,  step=0.1)
        hemo = st.number_input('Hemoglobin',             min_value=3.0,  max_value=20.0,  value=13.0, step=0.1)
        pcv  = st.number_input('Packed Cell Volume',     min_value=10,   max_value=60,    value=40)
        wc   = st.number_input('White Blood Cell Count', min_value=2000, max_value=20000, value=8000)
        rc   = st.number_input('Red Blood Cell Count',   min_value=2.0,  max_value=8.0,   value=4.5,  step=0.1)

    with col3:
        st.markdown('**🔤 Categorical Features**')
        rbc   = st.selectbox('Red Blood Cells',          ['normal', 'abnormal'])
        pc    = st.selectbox('Pus Cell',                 ['normal', 'abnormal'])
        pcc   = st.selectbox('Pus Cell Clumps',          ['notpresent', 'present'])
        ba    = st.selectbox('Bacteria',                 ['notpresent', 'present'])
        htn   = st.selectbox('Hypertension',             ['no', 'yes'])
        dm    = st.selectbox('Diabetes Mellitus',        ['no', 'yes'])
        cad   = st.selectbox('Coronary Artery Disease',  ['no', 'yes'])
        appet = st.selectbox('Appetite',                 ['good', 'poor'])
        pe    = st.selectbox('Pedal Edema',              ['no', 'yes'])
        ane   = st.selectbox('Anemia',                   ['no', 'yes'])

    st.divider()

    # ── Predict button ────────────────────────────────────────────
    if st.button('🔍 Run Prediction'):

        # Encode categoricals
        cat_map = {
            'rbc'  : {'normal': 1, 'abnormal': 0},
            'pc'   : {'normal': 1, 'abnormal': 0},
            'pcc'  : {'notpresent': 0, 'present': 1},
            'ba'   : {'notpresent': 0, 'present': 1},
            'htn'  : {'no': 0, 'yes': 1},
            'dm'   : {'no': 0, 'yes': 1},
            'cad'  : {'no': 0, 'yes': 1},
            'appet': {'good': 0, 'poor': 1},
            'pe'   : {'no': 0, 'yes': 1},
            'ane'  : {'no': 0, 'yes': 1},
        }

        # Build input in exact training column order
        input_data = pd.DataFrame([{
            'age'  : age,   'bp'  : bp,    'sg'  : sg,   'al'  : al,
            'su'   : su,
            'rbc'  : cat_map['rbc'][rbc],
            'pc'   : cat_map['pc'][pc],
            'pcc'  : cat_map['pcc'][pcc],
            'ba'   : cat_map['ba'][ba],
            'bgr'  : bgr,   'bu'  : bu,    'sc'  : sc,
            'sod'  : sod,   'pot' : pot,   'hemo': hemo,
            'pcv'  : pcv,   'wc'  : wc,    'rc'  : rc,
            'htn'  : cat_map['htn'][htn],
            'dm'   : cat_map['dm'][dm],
            'cad'  : cat_map['cad'][cad],
            'appet': cat_map['appet'][appet],
            'pe'   : cat_map['pe'][pe],
            'ane'  : cat_map['ane'][ane],
        }])

        # Scale numerical columns
        input_scaled = input_data.copy()
        input_scaled[numerical_cols] = scaler.transform(input_data[numerical_cols])

        # Predict
        prediction  = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        risk_pct    = probability * 100

        # ── Result display ────────────────────────────────────────
        st.subheader('🧾 Prediction Result')

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown('<div class="metric-card">'
                        f'<h3>Diagnosis</h3>'
                        f'<h2>{"⚠️ CKD" if prediction == 1 else "No CKD"}</h2>'
                        '</div>', unsafe_allow_html=True)
        with r2:
            st.markdown('<div class="metric-card">'
                        f'<h3>CKD Probability</h3>'
                        f'<h2>{risk_pct:.1f}%</h2>'
                        '</div>', unsafe_allow_html=True)
        with r3:
            risk_level = 'High 🔴' if risk_pct >= 70 else ('Medium 🟡' if risk_pct >= 40 else 'Low 🟢')
            st.markdown('<div class="metric-card">'
                        f'<h3>Risk Level</h3>'
                        f'<h2>{risk_level}</h2>'
                        '</div>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)

        if prediction == 1:
            st.markdown('<div class="result-box ckd-box">⚠️ CKD Detected — Please refer to a nephrologist immediately.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box no-ckd-box">No CKD Detected — Patient appears healthy. Recommend routine monitoring.</div>',
                        unsafe_allow_html=True)

        # ── Risk probability bar ──────────────────────────────────
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('**CKD Risk Meter:**')
        st.progress(int(risk_pct))

        # ── SHAP Feature Contribution Chart ──────────────────────
        st.divider()
        st.subheader('🔬 Feature Contribution to This Prediction (SHAP)')
        st.markdown('This chart shows **why** the model made this prediction — which features pushed toward or away from CKD.')

        with st.spinner('Generating SHAP explanation...'):
            explainer   = shap.TreeExplainer(model)
            shap_vals   = explainer.shap_values(input_scaled)

            # Handle both old and new SHAP formats
            if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                sv       = shap_vals[:, :, 1]
                base_val = explainer.expected_value[1]
            elif isinstance(shap_vals, list):
                sv       = shap_vals[1]
                base_val = explainer.expected_value[1]
            else:
                sv       = shap_vals
                base_val = explainer.expected_value

            # Waterfall plot for this patient
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(
                shap.Explanation(
                    values        = sv[0],
                    base_values   = base_val,
                    data          = input_scaled.iloc[0],
                    feature_names = input_scaled.columns.tolist()
                ),
                show=False
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown(
            '> 🔴 **Red bars** push the prediction **toward CKD**. '
            '🔵 **Blue bars** push it **away from CKD**.'
        )

        # ── Feature values summary ────────────────────────────────
        with st.expander('📊 View Full Input Summary'):
            summary_df = pd.DataFrame({
                'Feature' : training_col_order,
                'Value'   : [input_data[col].values[0] for col in training_col_order]
            })
            st.dataframe(summary_df, use_container_width=True)

        # ── Save to history ───────────────────────────────────────
        st.session_state.history.append({
            'Age'        : age,
            'BP'         : bp,
            'Hemoglobin' : hemo,
            'Creatinine' : sc,
            'Diagnosis'  : '⚠️ CKD' if prediction == 1 else 'No CKD',
            'Probability': f'{risk_pct:.1f}%',
            'Risk Level' : risk_level
        })
        st.success('Prediction saved to Patient History!')


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — PATIENT HISTORY
# ══════════════════════════════════════════════════════════════════
elif page == '📋 Patient History':

    st.title('📋 Patient Prediction History')
    st.markdown('All predictions made in this session are saved here.')
    st.divider()

    if len(st.session_state.history) == 0:
        st.info('No predictions yet. Go to **Home & Predict** to make your first prediction.')
    else:
        history_df = pd.DataFrame(st.session_state.history)
        history_df.index = history_df.index + 1
        history_df.index.name = 'Patient #'

        # Summary metrics
        total    = len(history_df)
        ckd_cnt  = history_df['Diagnosis'].str.contains('CKD ').sum()
        safe_cnt = total - ckd_cnt

        m1, m2, m3 = st.columns(3)
        m1.metric('Total Patients', total)
        m2.metric('CKD Detected',   ckd_cnt)
        m3.metric('No CKD',         safe_cnt)

        st.divider()
        st.dataframe(history_df, use_container_width=True)

        # Download as CSV
        csv = history_df.to_csv().encode('utf-8')
        st.download_button(
            label     = '⬇️ Download History as CSV',
            data      = csv,
            file_name = 'ckd_patient_history.csv',
            mime      = 'text/csv',
            use_container_width=True
        )

        # Clear history
        if st.button('🗑️ Clear History', use_container_width=True):
            st.session_state.history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT & HOW TO USE
# ══════════════════════════════════════════════════════════════════
elif page == '📖 About & How to Use':

    st.title('📖 About This App')
    st.divider()

    st.markdown("""
    ### 🔬 What is this app?
    This app uses a **Random Forest machine learning model** trained on the
    UCI Chronic Kidney Disease dataset to predict whether a patient has CKD
    based on 24 clinical features.

    ---
    ### 🧪 How to Use
    1. Go to **Home & Predict** from the sidebar
    2. Enter the patient's clinical values in the form
    3. Click **🔍 Run Prediction**
    4. View the diagnosis, risk probability, and SHAP explanation
    5. Check **📋 Patient History** to review all past predictions

    ---
    ### 📊 Model Performance
    | Metric    | Score  |
    |-----------|--------|
    | Accuracy  | 100%   |
    | Precision | 100%   |
    | Recall    | 100%   |
    | F1-Score  | 100%   |
    | AUC-ROC   | 100%   |

    ---
    ### 🧬 Key Features Used
    | Feature           | Clinical Significance                          |
    |-------------------|------------------------------------------------|
    | Hemoglobin        | Low levels indicate anemia common in CKD       |
    | Serum Creatinine  | High levels indicate poor kidney filtration    |
    | Packed Cell Volume| Low PCV reflects reduced red blood cell count  |
    | Specific Gravity  | Low SG indicates kidneys can't concentrate urine|
    | Blood Urea        | High levels suggest kidney failure             |

    ---
    ### ⚠️ Disclaimer
    This application is developed for **academic and research purposes only**.
    It should **not** be used as a substitute for professional medical diagnosis.
    Always consult a qualified nephrologist or physician for medical advice.

    ---
    ### 👨‍💻 Developer Info
    **Developed by Ani Merit(Final Year BME Student, UNN)**
    \n
            Supervised by Engr. Dr. Ojike
    \n
    - **Project:** CKD Prediction Using Machine Learning
    - **Dataset:** UCI CKD Dataset (via Kaggle)
    - **Model:** Random Forest Classifier
    - **Tools:** Python, Scikit-learn, SHAP, Streamlit
    """)
