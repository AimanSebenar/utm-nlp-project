# import streamlit as st
# from streamlit_option_menu import option_menu

# st.set_page_config(
#     page_title="Emotion Analyser",
#     page_icon="🧠",
#     layout="wide"
# )

# st.title("🧠 Social Media Emotion Analyser")
# st.caption("Predict emotion from social media text using ML models!")

# with st.sidebar:
#     selected = option_menu(
#         "Navigation",
#         ["Home", "Compare Models", "Analytics", "About"],
#         icons=["house", "cpu", "graph-up", "info-circle"]
#     )

# custom_text = st.text_area(
#     "Or enter your own text"
# )

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from streamlit_option_menu import option_menu
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Emotion AI Dashboard",
    page_icon="🧠",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.block-container {
    padding-top: 2rem;
}
.metric-card {
    border-radius: 12px;
    padding: 20px;
    background: #1E1E1E;
}
div[data-testid="metric-container"] {
    background-color: #262730;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TRAIN DATASET
# =========================
@st.cache_data
def load_training_data():
    base = os.path.dirname(__file__)
    train_path = os.path.join(base, "data", "train_cleaned.csv")
    train_df = pd.read_csv(train_path)

    if "text" not in train_df.columns or "emotion" not in train_df.columns:
        raise ValueError("Training data must contain 'text' and 'emotion' columns.")

    train_df = train_df[["text", "emotion"]].dropna().copy()
    train_df["emotion"] = train_df["emotion"].astype(str)

    sample_df = train_df.drop_duplicates(subset=["emotion"]).head(5).copy()

    return train_df, sample_df

train_df, sample_df = load_training_data()
sample_data = sample_df["text"].tolist()
emotions = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']

@st.cache_data
def load_model_info():
    base = os.path.dirname(__file__)
    comparison_path = os.path.join(base, "model_info", "model_comparison_results.csv")
    report_path = os.path.join(base, "model_info", "best_model_classification_report.csv")
    confusion_path = os.path.join(base, "model_info", "best_confusion_matrix.npy")

    comparison_df = pd.read_csv(comparison_path)
    report_df = pd.read_csv(report_path)
    confusion_matrix = np.load(confusion_path)

    report_df = report_df.copy()
    report_df.index = [
        "Sadness", "Joy", "Love", "Anger", "Fear", "Surprise", "Accuracy",
        "Macro Avg", "Weighted Avg"
    ]

    return comparison_df, report_df, confusion_matrix

comparison_df, report_df, confusion_matrix = load_model_info()

@st.cache_resource
def load_models():
    import os
    import joblib

    base = os.path.dirname(__file__)
    model_dir = os.path.join(base, "models")

    trad_path = os.path.join(model_dir, "model.pkl")
    adv_path = "C:/Users/Aiman/.vscode/nlp-project/models/roberta"
    vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

    trad_model = joblib.load(trad_path)

    tokenizer = RobertaTokenizer.from_pretrained(adv_path)
    adv_model = RobertaForSequenceClassification.from_pretrained(adv_path)
    adv_model.eval()
    adv_model.to("cpu")

    if adv_model is None:
        raise RuntimeError(
            "Could not load a Roberta transformer model from models/model.safetensors. "
            "Make sure model.safetensors and the local config/tokenizer files exist in the models/ directory."
        )

    vectorizer = joblib.load(vec_path)
    return trad_model, adv_model, vectorizer, tokenizer

trad_model, adv_model, vectorizer, tokenizer = load_models()

# =========================
# MOCK MODEL FUNCTIONS
# Replace with your actual model inference
# =========================
def predict_model1(text):
    start = time.time()

    transformed = vectorizer.transform([text])

    pred_id = trad_model.predict(transformed)[0]

    probs = trad_model.predict_proba(transformed)[0]

    latency = round(time.time() - start, 3)

    label_map = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise' }

    pred = label_map.get(pred_id, f"Unknown Label ID: {pred_id}")

    return pred, probs, latency

def predict_model2(text):
    start = time.time()

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    # Move inputs to CPU
    inputs = {k: v.to('cpu') for k, v in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = adv_model(**inputs)

    # Get predicted probabilities and class
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    predicted_class_id = torch.argmax(probs, dim=1).item()

    latency = round(time.time() - start, 3)

    label_map = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise' }

    pred = label_map.get(predicted_class_id, f"Unknown Label ID: {predicted_class_id}")

    return pred, probs.tolist(), latency

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Text Analyzer", "Dataset Explorer", "Visualizations", "Model Info"],
        icons=["house", "alphabet-uppercase", "database", "graph-up","info-circle"],
        default_index=0
    )

# =========================
# HEADER
# =========================
st.title("Social Media Emotion Analyzer (Group Doomsday)")
st.caption("Predict emotion from social media texts.")

# =========================
# HOME PAGE
# =========================
if selected == "Home":
    st.subheader("Welcome to the Emotion Analyzer")

    st.markdown("""
    This app is designed to help us understand the emotional tone of social media text.
    Social media posts often contain hidden emotions such as joy, sadness, fear, anger, love, and surprise,
    but reading and classifying them manually can be slow and inconsistent.
    """)

    st.markdown("### What problem are we solving?")
    st.markdown("""
    - Many online conversations contain emotional signals that are hard to interpret at scale.
    - Businesses, researchers, and moderators need a faster way to identify the emotions behind text.
    - Our project uses NLP and machine learning to classify text automatically and make this process more efficient.
    """)

    st.markdown("### How to use the app")
    st.markdown("""
    1. Open the Text Analyzer page from the sidebar.
    2. Enter your own text or choose one of the sample messages.
    3. Click the Run Prediction button.
    4. Compare the predictions from the two models and review the confidence scores.
    5. Explore the dataset and analytics pages for more insights.
    """)

    st.markdown("### Group Members")
    st.markdown("""
    - Member 1: Full Name — Matric Number
    - Member 2: Full Name — Matric Number
    - Member 3: Full Name — Matric Number
    """)

# =========================
# COMPARE MODELS PAGE
# =========================
elif selected == "Text Analyzer":

    st.subheader("Input Text")

    colA, colB = st.columns([2,1])

    with colA:
        custom_text = st.text_area("Enter custom text")

    with colB:
        sample_text = st.selectbox("Or choose sample text", sample_data)

    input_text = custom_text if custom_text else sample_text

    st.caption(f"Character count: {len(input_text)}")

    if st.button("Run Prediction", use_container_width=True):

        # MODEL INFERENCE
        pred1, probs1, latency1 = predict_model1(input_text)
        pred2, probs2, latency2 = predict_model2(input_text)
        probs2 = [item for sublist in probs2 for item in sublist]

        # =========================
        # MODEL OUTPUT CARDS
        # =========================
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Logistic Regression")
            st.metric("Prediction", pred1)
            st.metric("Confidence", f"{max(probs1)*100:.2f}%")
            st.metric("Latency", f"{latency1}s")

        with col2:
            st.subheader("RoBERTa")
            st.metric("Prediction", pred2)
            st.metric("Confidence", f"{max(probs2)*100:.2f}%")
            st.metric("Latency", f"{latency2}s")

        # =========================
        # AGREEMENT STATUS
        # =========================
        st.divider()

        if pred1 == pred2:
            st.success(f"✅ Both models agree → {pred1}")
        else:
            st.warning(f"⚠ Models disagree | Logistic Regression: {pred1} | RoBERTa: {pred2}")

        # =========================
        # PROBABILITY VISUALIZATION
        # =========================
        st.divider()
        st.subheader("Probability Distribution")

        col3, col4 = st.columns(2)

        df1 = pd.DataFrame({
            "Emotion": emotions,
            "Probability": probs1
        })

        df2 = pd.DataFrame({
            "Emotion": emotions,
            "Probability": probs2
        })

        with col3:
            fig1 = px.bar(
                df1,
                x="Emotion",
                y="Probability",
                title="Logistic Regression Distribution"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col4:
            fig2 = px.bar(
                df2,
                x="Emotion",
                y="Probability",
                title="RoBERTa Distribution"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # =========================
        # EMOTION ICON DISPLAY
        # =========================
        emotion_icons = {
            "Joy": "😊",
            "Anger": "😡",
            "Sadness": "😢",
            "Surprise": "😲",
            "Fear": "😨",
            "Love": "❤️"
        }

        st.divider()
        col5, col6 = st.columns(2)

        with col5:
            st.markdown(f"### {emotion_icons[pred1]} Logistic Regression Emotion: **{pred1}**")

        with col6:
            st.markdown(f"### {emotion_icons[pred2]} RoBERTa Emotion: **{pred2}**")

# =========================
# ANALYTICS PAGE
# =========================
elif selected == "Analytics":

    st.subheader("Benchmark Metrics")

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Model 1": [0.88, 0.86, 0.84, 0.85],
        "Model 2": [0.92, 0.90, 0.91, 0.90]
    })

    st.dataframe(metrics, use_container_width=True)

    fig = px.bar(
        metrics,
        x="Metric",
        y=["Model 1", "Model 2"],
        barmode="group",
        title="Model Performance Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# DATASET EXPLORER
# =========================
elif selected == "Dataset Explorer":

    st.subheader("Dataset Samples")

    sample_display_df = pd.DataFrame({
        "Text": sample_df["text"].tolist(),
        "Emotion Label": sample_df["emotion"].tolist()
    })

    emotion_filter = st.multiselect(
        "Filter Emotion",
        emotions,
        default=emotions
    )

    filtered = sample_display_df[sample_display_df["Emotion Label"].isin(emotion_filter)]

    st.dataframe(filtered, use_container_width=True)

    distribution_df = pd.DataFrame({
        "Emotion": emotions,
        "Count": [int((train_df["emotion"] == emotion).sum()) for emotion in emotions]
    })

    fig = px.bar(
        distribution_df,
        x="Emotion",
        y="Count",
        title="Real Emotion Distribution in Training Data"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# VISUALIZATIONS PAGE
# =========================
elif selected == "Visualizations":

    st.subheader("Project Visualizations")

    visuals_dir = os.path.join(os.path.dirname(__file__), "visualisations")
    image_files = sorted([
        os.path.join(visuals_dir, file)
        for file in os.listdir(visuals_dir)
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))
    ])

    if not image_files:
        st.info("No visualization images were found in the visualisations folder.")
    else:
        cols = st.columns(2)
        for index, image_path in enumerate(image_files):
            title = os.path.splitext(os.path.basename(image_path))[0].replace("_", " ").title()
            with cols[index % 2]:
                st.image(image_path, caption=title, use_container_width=True)

# =========================
# MODEL INFO PAGE
# =========================
elif selected == "Model Info":

    st.subheader("Model Information and Performance")

    st.markdown("The model details below are loaded directly from the files in the model_info folder.")

    st.markdown("### Model Comparison Results")
    st.dataframe(comparison_df, use_container_width=True)

    best_model = comparison_df.sort_values("Test F1-Score", ascending=False).iloc[0]["Model Pipeline Configuration"]
    st.success(f"Best-performing model based on test F1-score: {best_model}")

    st.markdown("### Classification Report")
    st.dataframe(report_df, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{report_df.loc['Accuracy', 'precision'] * 100:.2f}%")
    with col2:
        st.metric("Macro Avg F1", f"{report_df.loc['Macro Avg', 'f1-score'] * 100:.2f}%")
    with col3:
        st.metric("Weighted Avg F1", f"{report_df.loc['Weighted Avg', 'f1-score'] * 100:.2f}%")

    st.markdown("### Confusion Matrix")
    fig = px.imshow(
        confusion_matrix,
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        x=emotions,
        y=emotions,
        title="Best Model Confusion Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# FOOTER
# =========================
st.divider()
st.caption("Built with Streamlit | Emotion Classification Model Benchmark Dashboard")