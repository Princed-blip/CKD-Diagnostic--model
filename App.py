import streamlit as st
import pickle
import pandas as pd
import numpy as np


# Load the model
model = pickle.load(open("ckd_model.pkl", 'rb')) 

# my information
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.image("C:/Users/HP/OneDrive/Pictures/Saved Pictures/IMG-20250522-WA0032.jpg", width=100)
    st.write("Contact: [ifeanyistephen003@gmail.com]")
st.write("---")


# Create a Streamlit app
st.title("Chronic Kidney Disease(CDK) Prediction App")

# Input fields
age = st.number_input("Age")
bp = st.number_input("Blood Pressure")
sg = st.number_input("Specific Gravity")
al = st.number_input("Albumin")
su = st.number_input("Sugar")
rbc = st.number_input("red blood cells")
pc = st.number_input("pus cell")
pcc = st.number_input("pus cell clumps")
ba = st.number_input("bacteria")
bgr = st.number_input("blood glucose random")
bu = st.number_input("blood urea")
sc = st.number_input("serum creatinine")
sod = st.number_input("sodium")
pot = st.number_input("potassium")
hemo = st.number_input("hemoglobin")
pcv = st.number_input("packed cell volume")
wc = st.number_input("white blood cell count")
rc = st.number_input("red blood cell count")
htn = st.number_input("hypertension")
dm = st.number_input("diabetes mellitus")
cad = st.number_input("coronary artery disease")
appet = st.number_input("appetite")
pe = st.number_input("pedal edema")
ane = st.number_input("anemia")


def make_prediction(features):
    prediction = model.predict(features)
    return prediction

# Create a button to make predictions
if st.button("Predict"):
    features = pd.DataFrame([[age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]])  
    prediction = make_prediction(features)
    if prediction[0] == 0:
        st.write("This patient does not have CKD.")
    else:
        st.write("This patient has CKD.")   

st.write("---")

st.markdown("© 2025 All rights reserved | Developed by PRINCEDEX")


