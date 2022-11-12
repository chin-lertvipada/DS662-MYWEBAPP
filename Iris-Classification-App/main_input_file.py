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

st.sidebar.header("Input Features")

upload_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if upload_file:
    input_df = pd.read_csv(upload_file)
    X_test = input_df[["sepal.length","sepal.width","petal.length","petal.width"]]

# sepal_length = float(st.sidebar.slider("sepal length (cm)", 0.0, 10.0, 1.0))
# sepal_width = float(st.sidebar.slider("sepal width (cm)", 0.0, 7.0, 1.0))
# petal_length = float(st.sidebar.slider("petal length (cm)", 0.0, 10.0, 1.0))
# petal_width = float(st.sidebar.slider("petal width (cm)", 0.0, 7.0, 1.0))

btn = st.sidebar.button("Predict")

if btn:
    pred = model.predict(X_test)

    X_test['predict'] = pd.DataFrame(pred)
    
    st.dataframe(X_test)
    
    # pred = model.predict(np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(-1,4))

    # st.subheader(pred[0])
    # if pred[0]=='Setosa':
    #     st.image("Setosa.png")
    # elif pred[0] == 'Versicolor':
    #     st.image("Versicolor.png")
    # else:
    #     st.image("Virginica.png")

    # st.image(pred[0]+".png", caption=pred[0])


# run the app
# streamlit run main.py