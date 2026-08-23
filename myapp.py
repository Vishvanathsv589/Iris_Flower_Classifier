import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()

# Train model
model = RandomForestClassifier()
model.fit(iris.data, iris.target)

# Streamlit app
st.title("🌸 Iris Flower Classifier")

# Get input
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# Predict
if st.button("Predict"):

    data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(data)

    st.success(
        "Predicted Species: " +
        iris.target_names[prediction[0]]
    )
