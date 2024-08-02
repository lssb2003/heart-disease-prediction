import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

rbm = data["model"]

def show_predict_page():
    st.title("Heart Disease Predictor")



    age = st.slider("Age", 0, 100, 1)

    sex = st.selectbox("Sex", ("Male", "Female"))

    cp = st.selectbox("Chest pain type", ("typical angina", "atypical angina", "non-anginal pain", "asymptomatic"))

    trestbps = st.slider("Resting blood pressure in mm/Hg", 1, 300, 1)

    chol = st.slider("Serum cholesterol in mg/dl", 1, 350, 1)

    fbs = st.selectbox("Fasting blood suger > 120ml/dl", ("True", "False"))

    restecg = st.selectbox("Resting electrocardiographic results", ("normal", "having ST-T wave abnormality", "showing probable or definite left ventricular hypertrophy"))

    exang = st.selectbox("Exercise induced angina", ("Yes", "No"))

    ok = st.button("Make prediction")
    if ok:
        X = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, exang]])


        if X[:, 1] == "Male":
            X[:, 1] = 1
        else:
            X[:, 1] = 0



        if X[:, 2] == "typical angina":
            X[:, 2] = 1
        elif X[:, 2] == "atypical angina": 
            X[:, 2] = 2
        elif X[:, 2] == "non-anginal pain": 
            X[:, 2] = 3
        elif X[:, 2] == "asymptomatic":
            X[:, 2] = 4


        if X[:, 5] == "True":
            X[:, 5] = 1
        elif X[:, 5] == "False":
            X[:, 5] = 0
        

        if X[:, 6] == "normal":
            X[:, 6] = 0
        elif X[:, 6] == "having ST-T wave abnormality":
            X[:, 6] = 1
        elif X[:, 6] == "showing probable or definite left ventricular hypertrophy":
            X[:, 6] = 2

        
        if X[:, 7] == "Yes":
            X[:, 7] = 1
        elif X[:, 7] == "No":
            X[:, 7] = 0

        X = X.astype(float)


        res = rbm.predict(X)

        if res == 1:
            ans = "there is less than 50" + "%" + "diameter narrowing in any major blood vessel"
        elif res == 0:
            ans = "there is more than 50" + "%" + "diameter narrowing in any major blood vessels"
        


        st.subheader(f"{ans}")




        
