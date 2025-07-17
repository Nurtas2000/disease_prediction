"""
Disease Prediction System
Author: Nurtas Kalmakhan 
Date: 28.06.2025

A complete machine learning pipeline for predicting diseases (Diabetes, Heart Disease, Hypertension)
from health metrics including age, blood pressure, cholesterol, glucose, BMI, and smoking status.
"""

# ================ IMPORTS ================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, roc_curve, auc)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# ================ DATA LOADING ================
def load_data():
    """Load and return synthetic disease prediction dataset."""
    data = pd.DataFrame([
        [45,'Male',120,200,110,28.5,'Yes','Diabetes'],
        [50,'Female',140,240,95,32.0,'No','HeartDisease'],
        [38,'Male',110,180,85,24.0,'No','Healthy'],
        [60,'Female',150,280,140,35.5,'Yes','Diabetes'],
        [55,'Male',130,220,100,30.2,'Yes','HeartDisease'],
        [42,'Female',115,190,90,26.8,'No','Healthy'],
        [65,'Male',160,260,130,38.0,'Yes','Hypertension'],
        [48,'Female',125,210,105,29.7,'No','Diabetes'],
        [52,'Male',135,230,115,31.5,'Yes','HeartDisease'],
        [36,'Female',105,170,80,23.0,'No','Healthy'],
        [70,'Male',170,290,150,40.0,'Yes','Hypertension'],
        [44,'Female',118,195,88,25.5,'No','Healthy'],
        [58,'Male',145,250,120,33.8,'Yes','Diabetes'],
        [47,'Female',122,205,92,27.6,'No','Healthy'],
        [63,'Male',155,270,135,36.5,'Yes','Hypertension'],
        [41,'Female',112,185,87,24.8,'No','Healthy'],
        [56,'Male',138,235,110,32.5,'Yes','HeartDisease'],
        [49,'Female',128,215,98,28.9,'No','Diabetes'],
        [67,'Male',165,275,145,39.0,'Yes','Hypertension'],
        [39,'Female',108,175,82,22.5,'No','Healthy'],
        [53,'Male',142,225,118,31.0,'Yes','HeartDisease'],
        [46,'Female',119,198,89,26.0,'No','Healthy'],
        [62,'Male',158,265,138,37.2,'Yes','Hypertension'],
        [43,'Female',114,188,86,25.0,'No','Healthy'],
        [57,'Male',148,240,125,34.5,'Yes','Diabetes'],
        [51,'Female',132,218,102,30.0,'No','HeartDisease'],
        [69,'Male',168,285,148,41.0,'Yes','Hypertension'],
        [40,'Female',107,172,83,23.5,'No','Healthy'],
        [54,'Male',140,230,112,32.8,'Yes','Diabetes'],
        [59,'Female',135,225,108,31.2,'No','HeartDisease'],
        [37,'Male',103,165,78,21.8,'No','Healthy'],
        [66,'Male',162,272,142,38.5,'Yes','Hypertension'],
        [48,'Female',124,208,94,27.3,'No','Healthy'],
        [61,'Male',152,258,128,35.8,'Yes','Diabetes'],
        [48,'Male',126,212,96,28.0,'No','Healthy'],
        [64,'Female',156,268,132,36.8,'Yes','Hypertension'],
        [48,'Female',121,202,91,26.5,'No','Healthy'],
        [71,'Male',172,295,155,42.0,'Yes','Hypertension'],
        [48,'Female',117,192,84,24.3,'No','Healthy']
    ], columns=['Age','Gender','BloodPressure','Cholesterol','Glucose','BMI','Smoker','Diagnosis'])
    
    return data

# ================ EDA & VISUALIZATION ================
def perform_eda(data):
    """Perform exploratory data analysis and visualization."""
    print("\n=== Dataset Overview ===")
    print(f"Shape: {data.shape}")
    print("\n=== Data Types ===")
    print(data.dtypes)
    print("\n=== Missing Values ===")
    print(data.isnull().sum())
    print("\n=== Class Distribution ===")
    print(data['Diagnosis'].value_counts())
    
    # Numeric features distribution
    numeric_features = ['Age', 'BloodPressure', 'Cholesterol', 'Glucose', 'BMI']
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(numeric_features, 1):
        plt.subplot(2, 3, i)
        sns.histplot(data[feature], kde=True)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig('numeric_distributions.png')
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_data = data[numeric_features]
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.show()
    
    # Boxplots by diagnosis
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(numeric_features, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x='Diagnosis', y=feature, data=data)
        plt.title(f'{feature} by Diagnosis')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('features_by_diagnosis.png')
    plt.show()

# ================ DATA PREPROCESSING ================
def preprocess_data(data):
    """Preprocess data for machine learning."""
    # Encode categorical variables
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data['Smoker'] = data['Smoker'].map({'Yes': 1, 'No': 0})
    
    # Encode target variable
    le = LabelEncoder()
    data['Diagnosis'] = le.fit_transform(data['Diagnosis'])
    
    # Split features and target
    X = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_features = ['Age', 'BloodPressure', 'Cholesterol', 'Glucose', 'BMI']
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    return X_train, X_test, y_train, y_test, le, scaler

# ================ MODEL TRAINING ================
def train_models(X_train, y_train):
    """Train and evaluate multiple classification models."""
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    # Hyperparameter grids for tuning
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        },
        'Logistic Regression': {
            'C': [0.1, 1, 10],
            'penalty': ['l2']
        }
    }
    
    best_models = {}
    
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    
    return best_models

# ================ MODEL EVALUATION ================
def evaluate_models(models, X_test, y_test, le):
    """Evaluate models on test set and generate reports."""
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        })
        
        # Classification report
        print(f"\n=== {name} Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        
        # Confusion matrix
        plt.figure(figsize=(6, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'{name.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.show()
        
        # ROC curve (for multiclass)
        plt.figure(figsize=(8, 6))
        for i in range(len(le.classes_)):
            fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{le.classes_[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} ROC Curves')
        plt.legend(loc="lower right")
        plt.savefig(f'{name.lower().replace(" ", "_")}_roc_curve.png')
        plt.show()
    
    # Compare all models
    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison ===")
    print(results_df.sort_values('Accuracy', ascending=False))
    
    return results_df

# ================ MAIN EXECUTION ================
if __name__ == "__main__":
    print("=== Disease Prediction System ===")
    
    # Load data
    data = load_data()
    print("\nData loaded successfully!")
    
    # Perform EDA
    print("\nPerforming EDA...")
    perform_eda(data)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, le, scaler = preprocess_data(data)
    
    # Train models
    print("\nTraining models...")
    models = train_models(X_train, y_train)
    
    # Evaluate models
    print("\nEvaluating models...")
    results = evaluate_models(models, X_test, y_test, le)
    
    # Save best model
    best_model_name = results.loc[results['Accuracy'].idxmax(), 'Model']
    best_model = models[best_model_name]
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print(f"\nSaved best model ({best_model_name}) to 'best_model.pkl'")
    
    print("\n=== Analysis Complete ===")
