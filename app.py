import gradio as gr
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1️⃣ Load dataset and train a simple model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 2️⃣ Define prediction function
def predict(sepal_length, sepal_width, petal_length, petal_width):
    preds = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return iris.target_names[preds[0]]

# 3️⃣ Create Gradio interface
inputs = [
    gr.Number(label="Sepal length (cm)"),
    gr.Number(label="Sepal width (cm)"),
    gr.Number(label="Petal length (cm)"),
    gr.Number(label="Petal width (cm)")
]
output = gr.Textbox(label="Predicted Iris Species")

demo = gr.Interface(fn=predict, inputs=inputs, outputs=output,
                    title="🌸 Iris Flower Classifier",
                    description="Predicts the Iris species from flower dimensions.")

# 4️⃣ Launch app
if __name__ == "__main__":
    demo.launch()
