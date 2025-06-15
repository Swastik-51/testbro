# 🚨 CredHire AI: Job Fraud Detection System

[![Python](https://img.shields.io/badge/Built%20With-Python-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Deploy-Render](https://img.shields.io/badge/Deployed%20on-Render-46a2f1?logo=render)](https://credihireai.onrender.com)

---

## 📊 Project Overview
CredHire AI is a machine learning-based system that detects **fraudulent job postings** and protects job seekers from scams. It analyzes key job listing details and classifies them as either genuine or fraudulent, along with a confidence score.

🔗 **Live App**: [Streamlit Dashboard](https://cred-hire-ai.streamlit.app/)  
📡 **API Endpoint**: [FastAPI on Render](https://credihireai.onrender.com)  
🎥 **Demo Video**: [YouTube Demo](https://youtu.be/9lRWjJRMa6E)

---

## 🎯 Key Features

- 🧠 **Real-time job fraud detection**
- 📈 **Confidence score with every prediction**
- 📂 **CSV upload support for batch processing**
- 📊 **Interactive charts for analysis**
- 📤 **Email alert system** (in development)
- 💬 **SHAP explanations** for interpretability (coming soon)

---

## 🛠️ Technologies Used
- **Python**: Core programming language
- **Machine Learning**: 
  - Scikit-learn for model training
  - SHAP for model interpretability
  - XGBoost for gradient boosting
- **Data Processing**: 
  - Pandas for data manipulation
  - NLTK for text processing
- **Visualization**: 
  - Plotly for interactive charts
  - Dash for dashboard creation
- **Web Framework**: Flask for API endpoints

## 🤖 Machine Learning Models
We evaluated multiple classification models to find the best performer:

### Models Tested
1. Linear Support Vector Classification
2. Gradient Boosting
3. Gaussian Naive Bayes
4. Logistic Regression
5. K-Nearest Neighbors (KNN)
6. Random Forest
7. XGBoost

### Data Balancing
To address class imbalance in the dataset, we implemented three different sampling techniques:
- Random Oversampling
- Random Undersampling
- SMOTE (Synthetic Minority Over-sampling Technique)

### Model Performance
After extensive evaluation and hyperparameter tuning:

1. **Initial Performance**:
   - Random Forest Classifier achieved F1 score of 0.76 with all features

2. **Optimized Performance**:
   - Random Forest Classifier achieved F1 score of 0.98 using:
     - Selected important features
     - Tuned hyperparameters
     - Balanced dataset

### Final Model Selection
We selected the Random Forest Classifier as our production model due to its superior performance and robustness.

## 📋 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Swastik-51/CredHire-AI.git
   cd CredHire-AI
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the model files:
   - Model files are available at: [Google Drive Link]
   - Place `model.pkl` and `vectorizer.pkl` in the `models/` directory

### Running the Application
1. Start the dashboard:
   ```bash
   python app.py
   ```

2. Access the dashboard at: `http://localhost:8050`

## 📊 Model Performance
### Final Model Metrics
- Model: Random Forest Classifier
- F1 Score: 0.98
- Accuracy: 0.98
- Precision: 0.98
- Recall: 0.98
- Features: Selected important features
- Dataset: Balanced using sampling techniques

### Model Comparison Results
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Linear SVC | 0.85 | 0.84 |
| Gradient Boosting | 0.92 | 0.91 |
| Gaussian Naive Bayes | 0.88 | 0.87 |
| Logistic Regression | 0.86 | 0.85 |
| K-Nearest Neighbors | 0.89 | 0.88 |
| Random Forest | 0.98 | 0.98 |
| XGBoost | 0.95 | 0.94 |

The Random Forest Classifier outperformed all other models, achieving the highest accuracy and F1 score of 0.98, making it our chosen model for production deployment.

## 📁 Project Structure

```
CredHireAI/
├── app.py
├── config/
│   └── page_config.py
├── utils/
│   ├── model_loader.py
│   ├── shap_explainer.py
│   ├── email_sender.py
│   └── text_processing.py
├── sections/
│   ├── sidebar.py
│   ├── dashboard.py
│   ├── custom_input.py
│   └── api_server.py
├── models/
│   ├── model.pkl
│   └── ectorizer.pkl
├── jupyter notebook/
│   └── Fraud_Detection_final.ipynb
├── requirements.txt
```
- All model training, evaluation, and comparison details are available in the `jupyter notebook/` folder.

# 🛡️ CrediHireAI - Job Fraud Detection API

CrediHireAI is a machine learning-powered API that detects **fraudulent job postings** and helps protect job seekers from scams.

🔗 Live: [credihireai.onrender.com](https://credihireai.onrender.com)

---

## 🚀 Features

- Real-time fraud prediction
- XGBoost + TF-IDF + PCA
- Clean FastAPI backend
- Ready for integration with frontends (Streamlit, etc.)

---

## 📌 API Usage

### `POST /predict`

**Request:**
```json
{
  "title": "Software Engineer",
  "description": "Join our fast-growing tech team...",
  "company_profile": "XYZ Corp",
  "requirements": "Python, ML",
  "benefits": "Remote work"
}
```

## 🎥 Demo Video
https://youtu.be/9lRWjJRMa6E

## Working app
https://cred-hire-ai.streamlit.app/

## 👥 Team Members
- Swastik Sengupta - Team Lead
- Ahir Barman Maji - Team Member

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact
For any queries, please reach out to swastik724@gmail.com
