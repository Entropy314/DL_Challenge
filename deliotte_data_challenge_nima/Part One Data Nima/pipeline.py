import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def feature_engineer(dff:pd.DataFrame) -> pd.DataFrame:
    prefix = 'FEATURE'
    dff[f'{prefix}_ratio_pitch_note_density'] = dff[f'{prefix}_mean_pitch'] / dff[f'{prefix}_note_density']
    dff[f'{prefix}_ratio_pitch_duration'] = dff[f'{prefix}_mean_pitch'] / dff[f'{prefix}_mean_duration']
    # dff[f'{prefix}_ratio_pitch_tempo'] = dff[f'{prefix}_mean_pitch'] / dff[f'{prefix}_mean_tempo'] # Commented because it is giving me INF and NANs 
    dff[f'{prefix}_ratio_pitch_chord'] = dff[f'{prefix}_mean_pitch'] / dff[f'{prefix}_chord_diversity']
    return dff
    

def generate_synthetic_outliers(df_features, n_samples=5000):
    """
    Create synthetic outliers by randomizing known features.

    Parameters:
    - ps1_features: DataFrame of known composers' features.
    - n_samples: Number of synthetic outliers to generate.

    Returns:
    - DataFrame of synthetic outliers.
    """
    synthetic = df_features.sample(n=n_samples, replace=True).copy()
    for col in synthetic.columns:
        if synthetic[col].dtype != 'object':  # Only perturb numeric features
            synthetic[col] += np.random.normal(0, synthetic[col].std() * 2, size=n_samples)
    return synthetic


def preprocess_data(features:pd.DataFrame, target_column="label"):
    """
    Splits features and labels, and normalizes feature data.

    Parameters:
    - features: DataFrame with features and labels.
    - target_column: Column name containing labels.

    Returns:
    - X_train, X_test, y_train, y_test: Split and normalized datasets.
    """
    from sklearn.preprocessing import MinMaxScaler

    # Split features and labels
    X = features.drop(columns=[target_column])
    y = features[target_column]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Normalize features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def train_random_forest(X_train:np.array, y_train:np.array):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def train_one_class_svm(X_train:np.array):
    svm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.01)
    svm.fit(X_train)
    return svm

def train_isolation_forest(X_train:np.array):
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X_train)
    return iso_forest

def train_autoencoder(X_train:np.array, input_dim:int):
    # # # Autoencoder model
    autoencoder = models.Sequential([
        layers.InputLayer(shape=(input_dim,)),  # Expecting 2D input (batch_size, input_dim)
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(input_dim, activation='sigmoid'),
    ])
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X_train, X_train, epochs=100, batch_size=64, shuffle=True)
    return autoencoder

def evaluate_model(model, X_test:np.array, y_test:np.array, model_type:str="classification", threshold:float=0.5):
    """
    Evaluates the model and prints evaluation metrics.

    Parameters:
    - model: Trained model.
    - X_test: Test features.
    - y_test: Test labels.
    - model_type: Type of model ('classification', 'anomaly_detection', 'autoencoder').
    - threshold: Threshold for anomaly detection or reconstruction error.

    Returns:
    - None
    """
    if model_type == "classification":
        # Predict and evaluate classification models
        y_pred = model.predict(X_test)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    elif model_type == "anomaly_detection":
        # Predict for anomaly detection models
        y_scores = model.decision_function(X_test)
        
        y_pred = model.predict(X_test)
        
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("ROC-AUC Score:", roc_auc_score(y_test, -y_scores))  # Use negative scores

    elif model_type == "autoencoder":
        # Predict and compute reconstruction error
        reconstructed = model.predict(X_test)
        reconstruction_error = np.mean((X_test - reconstructed) ** 2, axis=1)
        y_pred = (reconstruction_error > threshold).astype(int)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("ROC-AUC Score:", roc_auc_score(y_test, -reconstruction_error))
        

def pipeline_with_synthetic_data(features:np.array, target_column:str="label", n_synthetic_samples:int=6000):
    # Split original data into train/test
    X_train, X_test, y_train, y_test, scaler = preprocess_data(features, target_column)
    

    print("Training One-Class SVM...")
    svm = train_one_class_svm(X_train)

    print("Training Isolation Forest...")
    iso_forest = train_isolation_forest(X_train)

    print("Training Autoencoder...")
    autoencoder = train_autoencoder(X_train, X_train.shape[1])
    
    if n_synthetic_samples: 
        # Generate synthetic data
        print("Generating synthetic data...")
        synthetic_data = generate_synthetic_outliers(pd.DataFrame(X_train, columns=features.drop(columns=[target_column]).columns), n_samples=n_synthetic_samples)
        synthetic_labels = np.zeros(len(synthetic_data))  # Synthetic data is labeled as outliers (0)
    
        synthetic_data_test = generate_synthetic_outliers(pd.DataFrame(X_test, columns=features.drop(columns=[target_column]).columns), n_samples=int(n_synthetic_samples * len(X_test)/(len(X_train)+len(X_test))))
        synthetic_labels_test = np.zeros(len(synthetic_data_test))
        
        # Add synthetic data to training set
        X_train = np.vstack([X_train, scaler.transform(synthetic_data)])
        y_train = np.hstack([y_train, synthetic_labels])
    
        # Add synthetic data to Validation set
        X_test = np.vstack([X_test, scaler.transform(synthetic_data_test)])
        y_test = np.hstack([y_test, synthetic_labels_test])
        # Train Models
        print("Training Random Forest...")
        rf = train_random_forest(X_train, y_train)
        
    # Evaluate Models

    print("\nEvaluating One-Class SVM...")
    evaluate_model(svm, X_test, y_test, model_type="anomaly_detection", threshold=0.5)

    print("\nEvaluating Random Forest...")
    evaluate_model(rf, X_test, y_test, model_type="classification")



    print("\nEvaluating Isolation Forest...")
    evaluate_model(iso_forest, X_test, y_test, model_type="anomaly_detection", threshold=0.0)

    print("\nEvaluating Autoencoder...")
    evaluate_model(autoencoder, X_test, y_test, model_type="autoencoder", threshold=0.95)
    data = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
    models = {'RF': rf, 'SVM': svm, 'ISO': iso_forest, 'AE': autoencoder, 'scaler': scaler}
    return data, models


def validate_models(features, target_column="label", n_synthetic_samples=50):
    """
    Validates models using labeled data.

    Parameters:
    - features: DataFrame with labeled features.
    - target_column: Column name containing labels.
    - n_synthetic_samples: Number of synthetic samples to generate.

    Returns:
    - Trained models: Random Forest, One-Class SVM, Isolation Forest, Autoencoder.
    - Scaler: Trained scaler for normalization.
    """
    # Preprocessing
    X_train, X_val, y_train, y_val, scaler = preprocess_data(features, target_column)

    if synthetic_data: 
        # Generate synthetic data
        print("Generating synthetic data...")
        synthetic_data = generate_synthetic_outliers(
            pd.DataFrame(X_train, columns=features.drop(columns=[target_column]).columns),
            n_samples=n_synthetic_samples,
        )
        synthetic_labels = np.zeros(len(synthetic_data))  # Synthetic data labeled as outliers (0)
    
        # Add synthetic data to training set
        X_train = np.vstack([X_train, scaler.transform(synthetic_data)])
        y_train = np.hstack([y_train, synthetic_labels])

    # Train Models
    print("Training Random Forest...")
    rf = train_random_forest(X_train, y_train)

    print("Training One-Class SVM...")
    svm = train_one_class_svm(X_train)

    print("Training Isolation Forest...")
    iso_forest = train_isolation_forest(X_train)

    print("Training Autoencoder...")
    autoencoder = train_autoencoder(X_train, X_train.shape[1])

    # Evaluate Models on Validation Set
    print("\nEvaluating Random Forest...")
    evaluate_model(rf, X_val, y_val, model_type="classification")
    plot_roc_curve(X_train, rf.predict_proba(X_train)[:1], title='Validation ROC Curve ')
    plot_roc_curve(X_val, rf.predict_proba(X_val)[:1], title='Validation ROC Curve ')

    plot_pr_curve(X_train, rf.predict_proba(X_train)[:1], title='Validation Precision-Recall Curve ')
    plot_pr_curve(X_val, rf.predict_proba(X_val)[:1], title='Validation Precision-Recall Curve ')

    print("\nEvaluating One-Class SVM...")
    evaluate_model(svm, X_val, y_val, model_type="anomaly_detection", threshold=0.0)

    print("\nEvaluating Isolation Forest...")
    evaluate_model(iso_forest, X_val, y_val, model_type="anomaly_detection", threshold=0.0)

    print("\nEvaluating Autoencoder...")
    evaluate_model(autoencoder, X_val, y_val, model_type="autoencoder", threshold=0.01)
    data = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
    models = {'RF': rf, 'SVM': svm, 'ISO': iso_forest, 'AE': autoencoder, 'scaler': scaler}
    return data, models

def predict_unlabeled_data(models, scaler, unlabeled_features):
    """
    Predicts labels for unlabeled test data using trained models.

    Parameters:
    - models: Tuple of trained models (Random Forest, One-Class SVM, Isolation Forest, Autoencoder).
    - scaler: Trained scaler for normalization.
    - unlabeled_features: DataFrame of unlabeled test data.

    Returns:
    - Dictionary of predictions from each model.
    """
    rf, svm, iso_forest, autoencoder = models

    # Normalize test features
    X_unlabeled = scaler.transform(unlabeled_features)

    # Predictions from Random Forest
    rf_preds = rf.predict(X_unlabeled)

    # Predictions from One-Class SVM
    svm_scores = svm.decision_function(X_unlabeled)
    svm_preds = (svm_scores < 0.0).astype(int)  # Outlier if score < 0

    # Predictions from Isolation Forest
    iso_forest_preds = iso_forest.predict(X_unlabeled)
    iso_forest_preds = (iso_forest_preds == -1).astype(int)  # Outlier if prediction == -1

    # Predictions from Autoencoder
    reconstructed = autoencoder.predict(X_unlabeled)
    reconstruction_error = np.mean((X_unlabeled - reconstructed) ** 2, axis=1)
    autoencoder_preds = (reconstruction_error > 0.01).astype(int)  # Threshold can be tuned

    return {
        "Random Forest": rf_preds,
        "One-Class SVM": svm_preds,
        "Isolation Forest": iso_forest_preds,
        "Autoencoder": autoencoder_preds,
    }


def full_pipeline(labeled_features, unlabeled_features, target_column="label", n_synthetic_samples=50):
    """
    Full pipeline for model validation on labeled data and inference on unlabeled data.

    Parameters:
    - labeled_features: DataFrame with labeled features and labels.
    - unlabeled_features: DataFrame of features for unlabeled data.
    - target_column: Column name containing labels in the labeled dataset.
    - n_synthetic_samples: Number of synthetic samples to generate for training.

    Returns:
    - Predictions for the unlabeled dataset from all models.
    """
    # Step 1: Validate models on labeled data
    print("Validating models on labeled data...")
    models, scaler = validate_models(labeled_features, target_column, n_synthetic_samples)[:5]

    # Step 2: Predict on unlabeled test data
    print("\nPredicting labels for unlabeled test data...")
    predictions = predict_unlabeled_data(models, scaler, unlabeled_features)

    # Display predictions
    for model_name, preds in predictions.items():
        print(f"{model_name} Predictions: {preds}")

    return predictions



def plot_roc_curve(y_true, y_scores, title="ROC Curve"):
    """
    Plots the ROC curve.

    Parameters:
    - y_true: Ground truth labels.
    - y_scores: Predicted scores or probabilities.
    - title: Title for the ROC curve plot.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def plot_pr_curve(y_true, y_scores, title="Precision-Recall Curve"):
    """
    Plots the Precision-Recall curve.

    Parameters:
    - y_true: Ground truth labels.
    - y_scores: Predicted scores or probabilities.
    - title: Title for the PR curve plot.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, color="blue", lw=2, label=f"PR Curve (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()


def visualize_feature_space(X, y, method="pca", title="Feature Space Visualization"):
    """
    Visualizes the feature space using PCA or t-SNE.

    Parameters:
    - X: Feature matrix.
    - y: Labels (inlier/outlier).
    - method: Dimensionality reduction method ('pca' or 'tsne').
    - title: Title for the plot.
    """
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")

    reduced_data = reducer.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(
        reduced_data[y == 1, 0],
        reduced_data[y == 1, 1],
        label="Inlier",
        alpha=0.1,
        c="blue",
    )
    plt.scatter(
        reduced_data[y == 0, 0],
        reduced_data[y == 0, 1],
        label="Outlier",
        alpha=0.1,
        c="red",
    )
    plt.title(title)
    plt.legend()
    plt.show()


