##### install streamlit
# $ pip install streamlit
# $ streamlit hello

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

model = pickle.load(open("iris-model.pickle", "rb"))

st.title("Iris Species Prediction")

sepal_length = float(st.number_input("sepal length (cm)"))
sepal_width = float(st.number_input("sepal width (cm)"))
petal_length = float(st.number_input("petal length (cm)"))
petal_width = float(st.number_input("petal width (cm)"))

btn = st.button("Predict")

if btn:
    pred = model.predict(np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(-1,4))

    st.subheader(pred[0])
    # if pred[0]=='Setosa':
    #     st.image("setosa.png")
    # elif pred[0] == 'Versicolor':
    #     st.image("versicolor.png")
    # else:
    #     st.image("virginica.png")

    st.image(pred[0]+".png", caption=pred[0])


# run the app
# streamlit run main.py