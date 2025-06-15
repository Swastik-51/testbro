import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# ---------------------- Page Config ----------------------
st.set_page_config(page_title="CrediHireAI - Job Scam Detector", layout="wide")
st.title("CrediHire AI – Credible hiring through AI.")
st.markdown("### Your AI-powered shield against job scams. Apply with confidence!")

# ---------------------- Load Model + Vectorizer ----------------------
import os
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource
def load_resources():
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "ahirr_vectorizer.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "ahirr_model.pkl"))
    return vectorizer, model


vectorizer, model = load_resources()

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.title("Upload Job Listings")
    uploaded_file = st.file_uploader("Choose CSV File", type=["csv"])
    st.markdown("---")
    st.caption("Built with love using ML and Streamlit")

# ---------------------- CSV-Based Analysis ----------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    with st.expander("Uploaded Data Preview", expanded=True):
        st.dataframe(df.head(), use_container_width=True)

    for col in ['title', 'company_profile', 'description', 'requirements', 'benefits']:
        df[col] = df.get(col, "")

    df['text'] = (df['title'].fillna('') + ' ' +
                  df['company_profile'].fillna('') + ' ' +
                  df['description'].fillna('') + ' ' +
                  df['requirements'].fillna('') + ' ' +
                  df['benefits'].fillna('')).str.lower().str.strip()

    X_sparse = vectorizer.transform(df['text'])
    probs = model.predict_proba(X_sparse)[:, 1]
    preds = (probs > 0.5).astype(int)

    df['Fraud Probability'] = probs
    df['Prediction'] = preds
    df['Prediction Label'] = df['Prediction'].map({0: 'Genuine', 1: 'Fraud'})

    # ---------------------- KPIs ----------------------
    st.subheader("Summary Metrics")
    total_jobs = len(df)
    total_frauds = df['Prediction'].sum()
    fraud_rate = (total_frauds / total_jobs) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Jobs Analyzed", f"{total_jobs:,}")
    col2.metric("Fraudulent Listings", f"{total_frauds:,}")
    col3.metric("Fraud Rate (%)", f"{fraud_rate:.2f} %")

    st.divider()

    # ---------------------- Visual Insights ----------------------
    st.subheader("Visual Insights")
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(df, x='Fraud Probability', nbins=20,
                            color='Prediction Label',
                            color_discrete_map={'Fraud': 'red', 'Genuine': 'green'},
                            title="Fraud Probability Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.pie(df, names='Prediction Label',
                      color='Prediction Label',
                      color_discrete_map={'Fraud': 'red', 'Genuine': 'green'},
                      title="Genuine vs Fraud Breakdown")
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------------- Top Fraud Listings ----------------------
    st.subheader("Top 10 Suspicious Job Listings")
    st.dataframe(df.sort_values("Fraud Probability", ascending=False)
                 .head(10)[["title", "location", "Fraud Probability"]],
                 use_container_width=True)

    # ---------------------- SHAP Explanation ----------------------
    st.subheader("Why These Were Flagged, SHAP Plots")

    try:
        top3_text = df['text'].iloc[:3].tolist()
        top3_vec = vectorizer.transform(top3_text)

        top3_dense = pd.DataFrame(top3_vec.toarray(), columns=vectorizer.get_feature_names_out())

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(top3_dense)[1]

        for i in range(3):
            st.markdown(f"Listing #{i + 1}: {df.iloc[i]['title']}")
            feature_names = top3_dense.columns
            shap_vals = shap_values[i]
            instance_data = top3_dense.iloc[i]

            top_indices = np.argsort(np.abs(shap_vals))[-10:]
            explanation = shap.Explanation(
                values=shap_vals[top_indices],
                base_values=explainer.expected_value[1],
                data=instance_data.iloc[top_indices].values,
                feature_names=feature_names[top_indices]
            )

            fig = plt.figure(figsize=(8, 4))
            shap.plots.bar(explanation, show=False)
            st.pyplot(fig)

    except Exception as e:
        st.warning(f"SHAP visualization error: {e}")

    # ---------------------- Email Feature ----------------------
    st.subheader("Get Notified")

    with st.form("email_form"):
        user_email = st.text_input("Enter your email to receive the fraud listing report:")
        submit_email = st.form_submit_button("Send Email")

    if submit_email:
        if user_email and "@" in user_email:
            try:
                import smtplib
                from email.mime.multipart import MIMEMultipart
                from email.mime.text import MIMEText

                fraud_df = df[df['Prediction'] == 1][['title', 'location', 'Fraud Probability']]
                html_table = fraud_df.to_html(index=False)

                msg = MIMEMultipart()
                msg['From'] = "scamjobdetection176@gmail.com"
                msg['To'] = user_email
                msg['Subject'] = "FraudScan Report - Suspicious Job Listings"

                body = f"""
                <h3>Dear User,</h3>
                <p>Here are the job listings flagged as <b>fraudulent</b>:</p>
                {html_table}
                <p>Stay safe,<br><b>FraudScan</b></p>
                """
                msg.attach(MIMEText(body, 'html'))

                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login("scamjobdetection176@gmail.com", "yvoglollyyojfaua")
                server.send_message(msg)
                server.quit()

                st.success(f"Report successfully sent to {user_email}!")

            except Exception as e:
                st.error(f"Failed to send email: {e}")
        else:
            st.warning("Please enter a valid email address.")

    # ---------------------- Full Results ----------------------
    with st.expander("Full Prediction Results", expanded=True):
        st.dataframe(df[["title", "location", "Fraud Probability", "Prediction Label"]],
                     use_container_width=True)

    # ---------------------- Download Button ----------------------
    st.download_button(
        label="Download Results as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='fraud_predictions.csv',
        mime='text/csv'
    )

else:
    st.info("You can upload a CSV from the sidebar to analyze multiple listings.")


# ---------------------- Custom Job Description Input (Always Visible) ----------------------
st.subheader("🔍 Classify a Custom Job Description")

with st.form("custom_form", clear_on_submit=True):
    custom_title = st.text_input("Job Title")
    custom_company = st.text_area("Company Profile (optional)")
    custom_description = st.text_area("Job Description")
    custom_requirements = st.text_area("Requirements")
    custom_benefits = st.text_area("Benefits (optional)")

    submitted = st.form_submit_button("Classify Job Posting")

if submitted:
    combined_text = f"{custom_title} {custom_company} {custom_description} {custom_requirements} {custom_benefits}".lower().strip()
    transformed = vectorizer.transform([combined_text])
    probability = model.predict_proba(transformed)[0][1]
    prediction = "Fraud" if probability > 0.5 else "Genuine"

    st.markdown(f"### Result: **{prediction}**")
    st.progress(probability)
    st.markdown(f"**Fraud Probability:** `{probability:.2f}`")


# ---------------------- REST API Mode (Optional) ----------------------
if False:
    from flask import Flask, request, jsonify
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict_api():
        data = request.json
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "Missing 'text' field"}), 400
        text_vectorized = vectorizer.transform([text.lower()])
        prob = model.predict_proba(text_vectorized)[0, 1]
        prediction = int(prob > 0.5)
        return jsonify({
            "probability": round(float(prob), 4),
            "prediction": prediction,
            "label": "Fraud" if prediction else "Genuine"
        })

    app.run(debug=True)
