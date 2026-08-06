import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1️⃣ Load dataset and train a simple model
@st.cache_resource
def train_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=42
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, iris

model, iris = train_model()

# 2️⃣ Page setup
st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")
st.title("🌸 Iris Flower Classifier")
st.write("Predicts the Iris species from flower dimensions.")

# 3️⃣ Inputs
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, value=5.1, step=0.1)
    petal_length = st.number_input("Petal length (cm)", min_value=0.0, value=1.4, step=0.1)
with col2:
    sepal_width = st.number_input("Sepal width (cm)", min_value=0.0, value=3.5, step=0.1)
    petal_width = st.number_input("Petal width (cm)", min_value=0.0, value=0.2, step=0.1)

# 4️⃣ Prediction
if st.button("Predict Species"):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)[0]
    species = iris.target_names[prediction]
    st.success(f"Predicted Iris Species: **{species}**")

    # Optional: show prediction probabilities
    probs = model.predict_proba(features)[0]
    st.write("Prediction confidence:")
    st.bar_chart(
        {name: prob for name, prob in zip(iris.target_names, probs)}
    )