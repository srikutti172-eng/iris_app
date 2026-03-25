import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Iris App", layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1 {
    text-align: center;
    background: linear-gradient(to right, #ff4b2b, #ff416c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stButton>button {
    background-color: #ff4b2b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.stSlider label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD DATA (NO CSV) --------------------
iris = load_iris()

X = pd.DataFrame(iris.data, columns=[
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
])

y = pd.Series(iris.target).map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

# -------------------- MODEL --------------------
model = RandomForestClassifier()
model.fit(X, y)

# -------------------- TITLE --------------------
st.markdown("<h1>🌸 Iris Flower Classification</h1>", unsafe_allow_html=True)
st.markdown("### 🌿 Enter flower measurements below")

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 About App")
st.sidebar.info("""
This app predicts the species of Iris flower based on input measurements.

Model: Random Forest  
Dataset: Built-in sklearn Iris dataset  
""")

# -------------------- INPUT UI --------------------
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("🌱 Sepal Length", 4.0, 8.0, 5.0)
    sepal_width = st.slider("🌱 Sepal Width", 2.0, 4.5, 3.0)

with col2:
    petal_length = st.slider("🌸 Petal Length", 1.0, 7.0, 4.0)
    petal_width = st.slider("🌸 Petal Width", 0.1, 2.5, 1.0)

st.markdown("---")

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Species"):
    features = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=X.columns
    )

    prediction = model.predict(features)

    st.markdown("### 🌼 Prediction Result")
    st.success(f"✨ Predicted Iris Species: {prediction[0]}")
    st.balloons()