# **Hospital Readmission Prediction**

A machine learning project that predicts 30-day readmission likelihood for diabetic patients. Built using **Amazon SageMaker** for model training and deployment, and a **Flask** microservice for real-time inference.

---

## **Table of Contents**
1. [Project Overview](#project-overview)  
2. [Data Source](#data-source)  
3. [Project Architecture](#project-architecture)  
4. [Technical Stack](#technical-stack)  
5. [Data Pipeline](#data-pipeline)  
6. [Modeling and Evaluation](#modeling-and-evaluation)  
7. [Deployment](#deployment)  
8. [Flask Inference API](#flask-inference-api)  
9. [Getting Started](#getting-started)  
10. [Repository Structure](#repository-structure)  
11. [Future Work](#future-work)  
12. [Contributors](#contributors)  

---

## **Project Overview**

**Goal**: Predict whether a diabetic patient will be readmitted within 30 days of hospital discharge.  
**Why It Matters**: Hospital readmissions are costly and often indicative of care quality. By identifying high-risk patients, healthcare providers can intervene early, personalize follow-up care, and reduce overall readmission rates.

**Key Features**:
- End-to-end pipeline: data ingestion, cleaning, feature engineering, model training, deployment, and real-time inference.  
- Hosted on **AWS SageMaker** for scalable model training and serving.  
- Uses a **Flask** API to demonstrate how predictions can be integrated into real-time hospital systems.  
- Showcases handling of **imbalanced classification**, typical in healthcare contexts.

---

## **Data Source**

We use the **Diabetes 130-US hospitals for years 1999-2008** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008).

- **Size**: ~100k records  
- **Features**: Demographics, admission details, medications, lab results, discharge info, and readmission status  
- **Target**: Binary indicator of readmission within 30 days

> **Note**: Due to data licensing and size constraints, the raw dataset is **not** included in this repo. Please download it separately and upload to an Amazon S3 bucket or refer to the instructions in the [Getting Started](#getting-started) section below.

---

## **Project Architecture**

Here's a high-level overview of the solution:

1. **Data Storage**: Raw CSV uploaded to **Amazon S3**.  
2. **Exploration & Preprocessing**: Done in **Amazon SageMaker Studio** notebooks for data cleaning and feature engineering.  
3. **Model Training**: Trained using **SageMaker** (XGBoost or another algorithm) with hyperparameter tuning.  
4. **Model Deployment**: Best model is deployed to a **SageMaker Endpoint**.  
5. **Inference Microservice**: A simple **Flask** app calls the SageMaker endpoint for real-time predictions on patient data.

```
Local or GitHub Repo --> S3 (Data) --> SageMaker Studio (EDA, Feature Eng.) --> SageMaker Training & Tuning --> Endpoint --> Flask API --> End Users
```

---

## **Technical Stack**

- **Programming Language**: Python 3.8+  
- **AWS**: Amazon S3, Amazon SageMaker, SageMaker Studio  
- **ML Libraries**: NumPy, Pandas, scikit-learn, XGBoost, (optional) SMOTE for imbalance  
- **Framework**: Flask for microservice API  
- **Visualization**: Matplotlib, Seaborn (for EDA and metric plots)  
- **Containerization** (Optional): Dockerfile if you plan to build custom containers  

---

## **Data Pipeline**

1. **Data Import**:  
   - Download the dataset from UCI.  
   - Upload to an **S3** bucket.  

2. **Data Cleaning**:  
   - Remove or impute missing values (e.g., “?” in the dataset).  
   - Ensure consistent formatting for categorical variables (e.g., diagnosis codes).

3. **Feature Engineering**:  
   - Convert readmission status to a binary label (readmitted within 30 days vs. not).  
   - Extract useful features from medication changes, diagnoses, and admission info.  
   - Handle imbalanced labels (class weighting or SMOTE).  

4. **Train-Validation-Test Split**:  
   - Typically 70/15/15 or 80/10/10.  

---

## **Modeling and Evaluation**

1. **Model Selection**  
   - **XGBoost** baseline (SageMaker’s built-in container).  
   - Optional exploration of Logistic Regression, Random Forest, or Deep Learning.  

2. **Hyperparameter Tuning**  
   - Automated search using **SageMaker Hyperparameter Tuning** jobs (e.g., random or Bayesian search).  

3. **Evaluation Metrics**  
   - **Accuracy**  
   - **Precision** & **Recall** (vital in healthcare; balancing false positives vs. false negatives)  
   - **F1-Score**  
   - **ROC-AUC**  

4. **Results**  
   - (Include your final metrics here, confusion matrix, ROC curve, PR curve, etc.)

---

## **Deployment**

1. **Endpoint Creation**  
   - The best performing model is deployed to a **real-time endpoint** in SageMaker.  
   - You’ll need an `inference.py` script if you’re using a custom container or rely on XGBoost’s default if you’re using the built-in container.

2. **Endpoint Testing**  
   - Validate using `boto3` calls or Python scripts to ensure correct JSON in/out for predictions.

---

## **Flask Inference API**

- **File**: [flask_app/app.py](flask_app/app.py)  
- **Purpose**: Expose an endpoint `/predict_readmission` that accepts patient data (JSON) and returns a prediction.

### **Request Format**
```json
{
  "gender": "Male",
  "age": "[50-60)",
  "num_medications": 13,
  "diag_1": "250.02",
  "diag_2": "401.9",
  "insulin": "Down",
  ...
}
```

### **Response Format**
```json
{
  "readmission_prediction": 1,
  "probability": 0.89
}
```
- `readmission_prediction` is `1` if readmitted <30 days, `0` otherwise.  
- `probability` indicates model confidence (0-1).

---

## **Getting Started**

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/YOUR-USERNAME/hospital-readmission-prediction.git
   cd hospital-readmission-prediction
   ```

2. **Create a Python Virtual Environment (optional but recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Data Setup**  
   - Download the dataset from [UCI Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008).  
   - Upload it to your S3 bucket. Update the S3 path in your notebooks or config files as needed.

5. **Open Jupyter Notebooks**  
   - Access **SageMaker Studio** or run `jupyter notebook` locally to explore the notebooks in `notebooks/`.  
   - Follow the steps in `01_exploration.ipynb` and `02_cleaning_features.ipynb` to clean and engineer features.

6. **Model Training**  
   - Use `03_model_training.ipynb` (or a custom `train.py` script) to run SageMaker training jobs.  
   - You may need to configure AWS credentials and roles.

7. **Deployment**  
   - Deploy the trained model to a SageMaker endpoint using the instructions in the `model_training` notebook or a dedicated script.

8. **Run Flask App (Locally)**  
   ```bash
   cd flask_app
   python app.py
   ```
   By default, it runs on `http://127.0.0.1:5000/`. Test with a sample JSON request.

---

## **Repository Structure**

```bash
hospital-readmission-prediction/
├── data/
│   └── sample_data.csv
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_cleaning_features.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── train.py               # If using a custom training script
│   └── inference.py           # If using a custom container
├── flask_app/
│   ├── app.py                 # Flask API
│   └── requirements.txt
├── Dockerfile                 # Optional if building a custom container
├── requirements.txt           # Dependencies
└── README.md                  # You're here!
```

- **`data/`**: Contains a small sample (not the entire dataset).  
- **`notebooks/`**: Jupyter notebooks for EDA, feature engineering, model training.  
- **`src/`**: Python scripts for training, inference, or utility functions.  
- **`flask_app/`**: Contains the Flask microservice and related requirements.  
- **`Dockerfile`**: If you choose to build a custom Docker image for SageMaker.  

---

## **Future Work**

- **Expand Feature Engineering**: Integrate external data or more detailed medical codes.  
- **Try Advanced Models**: Experiment with deep learning approaches or multi-task learning for different readmission windows.  
- **Explainability Tools**: Add **SHAP** or **LIME** analysis to highlight the top factors driving readmission.  
- **Automate Pipeline**: Use **AWS Step Functions** or **SageMaker Pipelines** for end-to-end automation.  

---

## **Contributors**

- Anova Youngers
- Stephanie Tabares

---
