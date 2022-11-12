##### install streamlit
# $ pip install streamlit
# $ streamlit hello

import streamlit as st
import pandas as pd

df = pd.read_parquet("./ndvi_sample.parquet")
ndvi_col = df.columns[3:]

st.write("NDVI")

st.line_chart(df[ndvi_col].iloc[0])



# run the app
# streamlit run main.py
