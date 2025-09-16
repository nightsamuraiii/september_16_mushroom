import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_score,
    recall_score,
    accuracy_score,
)


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous?🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous?🍄")

    @st.cache_data
    def load_data():
        data = pd.read_csv("mushrooms.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache_data
    def split(df):
        y = df["type"]
        x = df.drop(columns=["type"])
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0
        )
        return x_train, x_test, y_train, y_test

    def plot_metrics(model, x_test, y_test, metrics_list, class_names):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(
                model, x_test, y_test, display_labels=class_names, ax=ax
            )
            st.pyplot(fig)

        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)

        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)

    # Load & split data
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ["edible", "poisonous"]

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier",
        ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"),
    )

    # Declare metrics multiselect once with unique key
    metrics = st.sidebar.multiselect(
        "What metrics to plot?",
        ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
        key="metrics",
    )

    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C"
        )
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma")

        if st.sidebar.button("Classify", key="classify_svm"):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            # Metrics
            st.write("Accuracy: ", round(accuracy_score(y_test, y_pred), 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred), 2))

            # Plots
            plot_metrics(model, x_test, y_test, metrics, class_names)

    elif classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR"
        )
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key="max_iter")

        if st.sidebar.button("Classify", key="classify_lr"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            # Metrics
            st.write("Accuracy: ", round(accuracy_score(y_test, y_pred), 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred), 2))

            # Plots
            plot_metrics(model, x_test, y_test, metrics, class_names)

    elif classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input(
            "Number of tree in the forest", 100, 5000, step=10, key="n_estimators"
        )
        max_depth = st.sidebar.number_input(
            "Max depth", 1, 20, step=1, key="max_depth"
        )

        bootstrap_str = st.sidebar.radio(
            "Bootstrap samples when building trees",
            ('True', 'False'),    
            key='bootstrap'
        )
        bootstrap = True if bootstrap_str == 'True' else False

        if st.sidebar.button("Classify", key="classify_rf"):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            # Metrics
            st.write("Accuracy: ", round(accuracy_score(y_test, y_pred), 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred), 2))

            # Plots
            plot_metrics(model, x_test, y_test, metrics, class_names)



    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)


if __name__ == "__main__":
    main()
