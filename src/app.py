"""
================================================================================
PHONE PRICE CLASSIFICATION - STREAMLIT WEB APPLICATION
================================================================================
Interactive web application for predicting phone price categories based on
specifications. Users can either upload a CSV file or manually input phone
specifications.

Features:
    - CSV file upload and batch prediction
    - Manual input form for single predictions
    - Real-time predictions with confidence scores
    - Results visualization and export
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import warnings
from pathlib import Path
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

# Page configuration
st.set_page_config(
    page_title="Phone Price Classifier",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)


def inject_global_css():
    """Inject custom CSS for better styling."""
    st.markdown("""
    <style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Tooltip styling - fix background and text color */
    .stTooltip {
        background-color: #2d3748 !important;
        color: #ffffff !important;
        border: 1px solid #4a5568 !important;
        border-radius: 8px !important;
        padding: 12px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stTooltip > div {
        background-color: #2d3748 !important;
        color: #ffffff !important;
    }
    
    /* Fix tooltip text visibility - multiple selectors for compatibility */
    [data-testid="stTooltip"] {
        background-color: #2d3748 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stTooltip"] > div {
        background-color: #2d3748 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stTooltip"] p,
    [data-testid="stTooltip"] span,
    [data-testid="stTooltip"] div {
        color: #ffffff !important;
    }
    
    /* Alternative tooltip fix for BaseWeb components */
    div[data-baseweb="tooltip"] {
        background-color: #2d3748 !important;
        color: #ffffff !important;
    }
    
    div[data-baseweb="tooltip"] > div {
        background-color: #2d3748 !important;
        color: #ffffff !important;
    }
    
    div[data-baseweb="tooltip"] p,
    div[data-baseweb="tooltip"] span {
        color: #ffffff !important;
    }
    
    /* Target Streamlit's tooltip container */
    .stTooltip [class*="tooltip"],
    .stTooltip [class*="Tooltip"] {
        background-color: #2d3748 !important;
        color: #ffffff !important;
    }
    
    /* Card styling */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-card h3 {
        margin-top: 0;
        color: white;
    }
    
    /* Metric card styling */
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    /* Info box styling - dark theme */
    .info-box {
        background: #1e3a5f;
        color: #e0f2fe;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    
    .info-box strong {
        color: #ffffff;
    }
    
    .info-box p,
    .info-box div,
    .info-box span {
        color: #e0f2fe;
    }
    
    /* Success box styling - dark theme */
    .success-box {
        background: #1e3a2e;
        color: #d1fae5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
    }
    
    .success-box strong {
        color: #ffffff;
    }
    
    .success-box p,
    .success-box div,
    .success-box span {
        color: #d1fae5;
    }
    
    /* Warning box styling - dark theme */
    .warning-box {
        background: #3a2e1e;
        color: #fef3c7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    
    .warning-box strong {
        color: #ffffff;
    }
    
    .warning-box p,
    .warning-box div,
    .warning-box span {
        color: #fef3c7;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Training animation styles - Enhanced */
    @keyframes pulse {
        0%, 100% { 
            opacity: 1; 
            transform: scale(1);
        }
        50% { 
            opacity: 0.7; 
            transform: scale(1.02);
        }
    }
    
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    @keyframes slideIn {
        from { 
            transform: translateX(-30px); 
            opacity: 0; 
        }
        to { 
            transform: translateX(0); 
            opacity: 1; 
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 5px rgba(102, 126, 234, 0.5),
                        0 0 10px rgba(102, 126, 234, 0.3),
                        0 0 15px rgba(102, 126, 234, 0.2);
        }
        50% {
            box-shadow: 0 0 10px rgba(102, 126, 234, 0.8),
                        0 0 20px rgba(102, 126, 234, 0.6),
                        0 0 30px rgba(102, 126, 234, 0.4);
        }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .training-step {
        animation: slideIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 12px;
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .training-step::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.2),
            transparent
        );
        animation: shimmer 2s infinite;
    }
    
    .training-step.completed {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #10b98120 0%, #05966920 100%);
        animation: fadeInUp 0.5s ease-out;
    }
    
    .training-step.completed::before {
        display: none;
    }
    
    .training-step.active {
        border-left-color: #3b82f6;
        background: linear-gradient(
            135deg,
            #3b82f625 0%,
            #2563eb25 50%,
            #3b82f625 100%
        );
        background-size: 200% 200%;
        animation: pulse 2s infinite, gradientShift 3s ease infinite;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .training-step h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .training-step p {
        margin: 0.5rem 0 0 0;
        color: #666;
        font-size: 0.95rem;
    }
    
    .training-icon {
        display: inline-block;
        font-size: 1.5rem;
        animation: bounce 1s ease-in-out infinite;
        margin-right: 0.5rem;
    }
    
    .training-step.active .training-icon {
        animation: spin 2s linear infinite, bounce 1s ease-in-out infinite;
    }
    
    .training-step.completed .training-icon {
        animation: none;
    }
    
    .metric-card-training {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card-training::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: spin 3s linear infinite;
    }
    
    .metric-card-training:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    .spinner-icon {
        display: inline-block;
        animation: spin 1s linear infinite;
        margin-right: 0.5rem;
        font-size: 1.2rem;
    }
    
    .progress-glow {
        position: relative;
        overflow: hidden;
    }
    
    .progress-glow::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: 30%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.4),
            transparent
        );
        animation: shimmer 1.5s infinite;
    }
    
    .success-checkmark {
        display: inline-block;
        animation: fadeInUp 0.5s ease-out, bounce 0.6s ease-out 0.5s;
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    
    .training-container {
        position: relative;
        padding: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .model-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
        animation: fadeInUp 0.5s ease-out;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .model-badge:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 15px rgba(102, 126, 234, 0.5);
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .floating-emoji {
        display: inline-block;
        animation: float 2s ease-in-out infinite;
        font-size: 2rem;
    }
    
    /* Sidebar toggle button styling */
    button[data-testid="baseButton-secondary"][key="sidebar_toggle"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50% !important;
        width: 50px !important;
        height: 50px !important;
        font-size: 1.5rem !important;
        padding: 0 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s !important;
    }
    
    button[data-testid="baseButton-secondary"][key="sidebar_toggle"]:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """
    Load trained model and required artifacts.
    
    Returns:
    --------
    tuple: (model, train_columns, train_df) or (None, None, None) if not found
    """
    try:
        model_path = '../results/final_model.pkl'
        columns_path = '../results/train_columns.pkl'
        train_data_path = '../data/train.csv'
        
        if not all(os.path.exists(p) for p in [model_path, columns_path, train_data_path]):
            return None, None, None
        
        model = joblib.load(model_path)
        train_columns = joblib.load(columns_path)
        train_df = pd.read_csv(train_data_path)
        
        return model, train_columns, train_df
    except Exception as e:
        st.error(f"Error loading artifacts: {str(e)}")
        return None, None, None


def predict_from_raw_df(raw_df: pd.DataFrame, model, train_columns, train_df_for_encoding):
    """
    Predict price categories from raw dataframe.
    
    Parameters:
    -----------
    raw_df : pandas.DataFrame
        Raw input dataframe with phone specifications
    model : trained model
        Trained ensemble model
    train_columns : list
        List of column names from training data
    train_df_for_encoding : pandas.DataFrame
        Training dataframe for proper encoding
        
    Returns:
    --------
    tuple: (predicted_labels, probabilities)
    """
    from preprocessing import preprocessing
    
    try:
        # Preprocess the input data
        processed_df = preprocessing(
            raw_df,
            is_train=False,
            train_df_for_encoding=train_df_for_encoding
        )
        
        # Align columns with training data
        processed_df = processed_df.reindex(columns=train_columns, fill_value=0)
        
        # Make predictions
        probabilities = model.predict_proba(processed_df)[:, 1]
        predictions = model.predict(processed_df)
        
        # Convert to labels
        pred_labels = ['expensive' if pred == 1 else 'non-expensive' for pred in predictions]
        
        return pred_labels, probabilities
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None


def train_model_with_progress(progress_bar, status_text, metrics_placeholder):
    """
    Train the ensemble model with progress tracking and visual feedback.
    
    Parameters:
    -----------
    progress_bar : streamlit.progress
        Progress bar widget
    status_text : streamlit.empty
        Status text placeholder
    metrics_placeholder : streamlit.empty
        Metrics display placeholder
    """
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    from preprocessing import preprocessing
    
    results = {}
    
    # Step 1: Load and preprocess data
    status_text.markdown("""
    <div class="training-step active">
    <h4><span class="training-icon">📊</span>Loading and Preprocessing Data...</h4>
    <p>Reading datasets and preparing features...</p>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(5)
    time.sleep(0.3)
    
    try:
        train_df = pd.read_csv('../data/train.csv')
        test_df = pd.read_csv('../data/test.csv')
    except FileNotFoundError:
        status_text.error("❌ Error: Could not find data files. Please ensure `data/train.csv` exists.")
        return None
    
    train_proc = preprocessing(train_df, is_train=True)
    test_proc = preprocessing(test_df, is_train=False, train_df_for_encoding=train_df)
    
    X_train = train_proc.drop('price', axis=1)
    y_train = train_proc['price']
    X_test = test_proc.reindex(columns=X_train.columns, fill_value=0)
    
    status_text.markdown(f"""
    <div class="training-step completed">
    <h4><span class="success-checkmark">✅</span>Data Loaded Successfully</h4>
    <p><strong>Training set:</strong> {X_train.shape[0]} samples, {X_train.shape[1]} features</p>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(10)
    time.sleep(0.2)
    
    # Step 2: Train Random Forest
    status_text.markdown("""
    <div class="training-step active">
    <h4><span class="training-icon">🌲</span>Training Random Forest Classifier...</h4>
    <p>Building decision trees and optimizing hyperparameters...</p>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(15)
    time.sleep(0.2)
    
    rf = RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    param_grid_rf = {
        'n_estimators': [900, 1000],
        'max_depth': [15, 18],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1],
    }
    
    grid_rf = RandomizedSearchCV(
        rf,
        param_grid_rf,
        n_iter=6,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        random_state=42
    )
    
    grid_rf.fit(X_train, y_train)
    results['rf'] = {
        'model': grid_rf.best_estimator_,
        'score': grid_rf.best_score_,
        'params': grid_rf.best_params_
    }
    
    status_text.markdown(f"""
    <div class="training-step completed">
    <h4><span class="success-checkmark">✅</span>Random Forest Complete</h4>
    <p><strong>Best CV Score:</strong> <span style="color: #10b981; font-weight: bold;">{grid_rf.best_score_:.4f}</span></p>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(30)
    time.sleep(0.2)
    
    # Step 3: Train Gradient Boosting
    status_text.markdown("""
    <div class="training-step active">
    <h4><span class="training-icon">📈</span>Training Gradient Boosting Classifier...</h4>
    <p>Sequentially building boosted trees with gradient descent...</p>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(35)
    time.sleep(0.2)
    
    gbdt = GradientBoostingClassifier(random_state=42)
    param_grid_gbdt = {
        'n_estimators': [500],
        'max_depth': [6],
        'learning_rate': [0.05],
        'subsample': [0.8],
        'min_samples_split': [5]
    }
    
    grid_gbdt = GridSearchCV(
        gbdt,
        param_grid_gbdt,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1
    )
    
    grid_gbdt.fit(X_train, y_train)
    results['gbdt'] = {
        'model': grid_gbdt.best_estimator_,
        'score': grid_gbdt.best_score_,
        'params': grid_gbdt.best_params_
    }
    
    status_text.markdown(f"""
    <div class="training-step completed">
    <h4><span class="success-checkmark">✅</span>Gradient Boosting Complete</h4>
    <p><strong>Best CV Score:</strong> <span style="color: #10b981; font-weight: bold;">{grid_gbdt.best_score_:.4f}</span></p>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(50)
    time.sleep(0.2)
    
    # Step 4: Train Neural Network
    status_text.markdown("""
    <div class="training-step active">
    <h4><span class="training-icon">🧠</span>Training Neural Network (MLP)...</h4>
    <p>Training deep learning model with backpropagation...</p>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(55)
    time.sleep(0.2)
    
    mlp_clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42
        )
    )
    
    param_grid_mlp = {
        'mlpclassifier__alpha': [1e-4],
        'mlpclassifier__learning_rate_init': [0.001]
    }
    
    grid_mlp = GridSearchCV(
        estimator=mlp_clf,
        param_grid=param_grid_mlp,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0
    )
    
    grid_mlp.fit(X_train, y_train)
    results['mlp'] = {
        'model': grid_mlp.best_estimator_,
        'score': grid_mlp.best_score_,
        'params': grid_mlp.best_params_
    }
    
    status_text.markdown(f"""
    <div class="training-step completed">
    <h4><span class="success-checkmark">✅</span>Neural Network Complete</h4>
    <p><strong>Best CV Score:</strong> <span style="color: #10b981; font-weight: bold;">{grid_mlp.best_score_:.4f}</span></p>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(65)
    time.sleep(0.2)
    
    # Step 5: Train CatBoost
    status_text.markdown("""
    <div class="training-step active">
    <h4><span class="training-icon">🐱</span>Training CatBoost Classifier...</h4>
    <p>Training gradient boosting optimized for categorical features...</p>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(70)
    time.sleep(0.2)
    
    cat_model = CatBoostClassifier(
        random_state=42,
        verbose=False,
        thread_count=-1,
        loss_function='Logloss',
        eval_metric='F1',
        iterations=700,
        depth=6,
        learning_rate=0.08,
        l2_leaf_reg=5,
        subsample=0.8,
        border_count=128,
        bagging_temperature=1.0
    )
    
    cat_model.fit(X_train, y_train)
    cat_score = cross_val_score(
        cat_model,
        X_train,
        y_train,
        cv=3,
        scoring='f1_weighted'
    ).mean()
    
    results['catboost'] = {
        'model': cat_model,
        'score': cat_score,
        'params': {}
    }
    
    status_text.markdown(f"""
    <div class="training-step completed">
    <h4><span class="success-checkmark">✅</span>CatBoost Complete</h4>
    <p><strong>CV Score:</strong> <span style="color: #10b981; font-weight: bold;">{cat_score:.4f}</span></p>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(75)
    time.sleep(0.2)
    
    # Display individual model scores with animations
    metrics_placeholder.markdown("""
    <div style="margin: 1.5rem 0;">
        <h3 style="text-align: center; margin-bottom: 1rem;">
            <span class="floating-emoji">📊</span> Individual Model Performance
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card-training" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🌲</div>
            <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">Random Forest</div>
            <div style="font-size: 1.5rem;">{results['rf']['score']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card-training" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📈</div>
            <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">Gradient Boosting</div>
            <div style="font-size: 1.5rem;">{results['gbdt']['score']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card-training" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧠</div>
            <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">Neural Network</div>
            <div style="font-size: 1.5rem;">{results['mlp']['score']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card-training" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🐱</div>
            <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">CatBoost</div>
            <div style="font-size: 1.5rem;">{results['catboost']['score']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Step 6: Calibrate models
    status_text.markdown("""
    <div class="training-step active">
    <h4><span class="training-icon">⚖️</span>Calibrating Models...</h4>
    <p>Improving probability estimates with isotonic calibration...</p>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(80)
    time.sleep(0.2)
    
    calibrated_rf = CalibratedClassifierCV(
        results['rf']['model'],
        method='isotonic',
        cv=3
    )
    calibrated_gb = CalibratedClassifierCV(
        results['gbdt']['model'],
        method='isotonic',
        cv=3
    )
    calibrated_mlp = CalibratedClassifierCV(
        results['mlp']['model'],
        method='isotonic',
        cv=3
    )
    calibrated_bonus = CalibratedClassifierCV(
        results['catboost']['model'],
        method='isotonic',
        cv=3
    )
    
    calibrated_rf.fit(X_train, y_train)
    calibrated_gb.fit(X_train, y_train)
    calibrated_mlp.fit(X_train, y_train)
    calibrated_bonus.fit(X_train, y_train)
    
    status_text.markdown("""
    <div class="training-step completed">
    <h4><span class="success-checkmark">✅</span>Model Calibration Complete</h4>
    <p>All models calibrated and ready for ensemble...</p>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(85)
    time.sleep(0.2)
    
    # Step 7: Optimize ensemble weights
    status_text.markdown("""
    <div class="training-step active">
    <h4><span class="training-icon">🎯</span>Optimizing Ensemble Weights...</h4>
    <p>Testing weight combinations to find optimal model blend...</p>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(87)
    time.sleep(0.2)
    
    weight_combinations = [
        [1, 5, 1, 2],
        [1, 6, 1, 2],
        [2, 5, 1, 2],
        [1, 5, 0, 3],
        [1, 6, 0, 3],
        [2, 5, 0, 3],
    ]
    
    best_weights = None
    best_ens_score = -1.0
    
    for i, weights in enumerate(weight_combinations):
        if weights[2] == 0:
            ens = VotingClassifier(
                estimators=[
                    ('rf', calibrated_rf),
                    ('gb', calibrated_gb),
                    ('bonus', calibrated_bonus)
                ],
                voting='soft',
                weights=[w for i, w in enumerate(weights) if i != 2]
            )
        else:
            ens = VotingClassifier(
                estimators=[
                    ('rf', calibrated_rf),
                    ('gb', calibrated_gb),
                    ('mlp', calibrated_mlp),
                    ('bonus', calibrated_bonus)
                ],
                voting='soft',
                weights=weights
            )
        
        scores = cross_val_score(ens, X_train, y_train, cv=3, scoring='f1_weighted')
        mean_score = scores.mean()
        
        if mean_score > best_ens_score:
            best_ens_score = mean_score
            best_weights = weights
        
        progress_bar.progress(87 + int((i + 1) * 3 / len(weight_combinations)))
    
    # Step 8: Create final ensemble
    status_text.markdown("""
    <div class="training-step active">
    <h4><span class="training-icon">🔮</span>Creating Final Ensemble Model...</h4>
    <p>Combining models with optimized weights...</p>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(95)
    time.sleep(0.2)
    
    if best_weights[2] == 0:
        final_model = VotingClassifier(
            estimators=[
                ('rf', calibrated_rf),
                ('gb', calibrated_gb),
                ('bonus', calibrated_bonus)
            ],
            voting='soft',
            weights=[w for i, w in enumerate(best_weights) if i != 2]
        )
        model_count = 3
    else:
        final_model = VotingClassifier(
            estimators=[
                ('rf', calibrated_rf),
                ('gb', calibrated_gb),
                ('mlp', calibrated_mlp),
                ('bonus', calibrated_bonus)
            ],
            voting='soft',
            weights=best_weights
        )
        model_count = 4
    
    final_model.fit(X_train, y_train)
    
    # Calculate validation accuracy (same as train.py)
    # Split data for validation (20% hold-out set)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        random_state=42,
        stratify=y_train,
        test_size=0.2
    )
    
    # Train on subset and predict on validation set
    final_model_val = VotingClassifier(
        estimators=final_model.estimators,
        voting=final_model.voting,
        weights=final_model.weights
    )
    final_model_val.fit(X_tr, y_tr)
    y_val_proba = final_model_val.predict_proba(X_val)[:, 1]
    
    # Optimize decision threshold for best F1-score
    best_thr = 0.5
    best_f1 = 0.0
    
    for thr in np.linspace(0.2, 0.8, 61):
        y_tmp = (y_val_proba >= thr).astype(int)
        f1 = f1_score(y_val, y_tmp, average='weighted')
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    
    # Make predictions with optimal threshold
    y_val_pred = (y_val_proba >= best_thr).astype(int)
    
    # Calculate accuracy
    final_accuracy = accuracy_score(y_val, y_val_pred)
    
    # Step 9: Save model
    status_text.markdown("""
    <div class="training-step active">
    <h4><span class="training-icon">💾</span>Saving Model and Artifacts...</h4>
    <p>Persisting trained model to disk...</p>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(98)
    time.sleep(0.2)
    
    os.makedirs('../results', exist_ok=True)
    joblib.dump(final_model, '../results/final_model.pkl')
    joblib.dump(X_train.columns.tolist(), '../results/train_columns.pkl')
    
    progress_bar.progress(100)
    
    weights_display = best_weights if model_count == 4 else [w for i, w in enumerate(best_weights) if i != 2]
    status_text.markdown(f"""
    <div class="training-step completed" style="background: linear-gradient(135deg, #10b98130 0%, #05966930 100%); border-left: 5px solid #10b981;">
        <h4 style="font-size: 1.5rem; margin-bottom: 1rem;">
            <span class="floating-emoji">🎉</span> Training Complete!
        </h4>
        <p style="font-size: 1.1rem; margin: 0.5rem 0;">
            <strong>Accuracy:</strong> 
            <span style="color: #10b981; font-weight: bold; font-size: 1.3rem;">{final_accuracy:.4f}</span>
        </p>
        <p style="margin: 0.5rem 0;">
            <strong>Best weights:</strong> 
            <span class="model-badge">{weights_display}</span>
        </p>
        <p style="margin-top: 1rem; color: #666;">
            Model saved and ready for predictions! 🚀
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    return {
        'model': final_model,
        'accuracy': final_accuracy,
        'weights': best_weights,
        'model_count': model_count,
        'train_columns': X_train.columns.tolist(),
        'train_df': train_df
    }


def manual_input_form():
    """
    Create manual input form for single phone prediction.
    
    Returns:
    --------
    pandas.DataFrame or None: Single-row dataframe with phone specs
    """
    st.markdown("### 📝 Enter Phone Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox(
            "Brand",
            ["Apple", "Samsung", "Xiaomi", "OnePlus", "Google", "Huawei", "Oppo", "Vivo", "Realme", "Other"],
            help="Select the phone brand"
        )
        
        performance_tier = st.selectbox(
            "Performance Tier",
            ["Entry-level", "Mid-range", "High-end", "Flagship"],
            help="Processor performance category"
        )
        
        ram_gb = st.selectbox(
            "RAM (GB)",
            [2, 3, 4, 6, 8, 12, 16, 18],
            index=3,
            help="Random Access Memory in gigabytes"
        )
        
        storage_gb = st.selectbox(
            "Storage (GB)",
            [32, 64, 128, 256, 512, 1024],
            index=2,
            help="Internal storage capacity"
        )
        
        battery_capacity = st.number_input(
            "Battery Capacity (mAh)",
            min_value=1000,
            max_value=10000,
            value=4000,
            step=100,
            help="Battery capacity in milliampere-hours"
        )
        
        screen_size = st.number_input(
            "Screen Size (inches)",
            min_value=4.0,
            max_value=8.0,
            value=6.5,
            step=0.1,
            help="Display screen size in inches"
        )
    
    with col2:
        refresh_rate = st.selectbox(
            "Refresh Rate",
            ["60 Hz", "90 Hz", "120 Hz", "144 Hz"],
            index=2,
            help="Screen refresh rate"
        )
        
        fast_charge_power = st.selectbox(
            "Fast Charging",
            ["No fast charging", "Up to 30W", "31–65W", "66W+"],
            index=1,
            help="Fast charging power capability"
        )
        
        rear_camera_mp = st.number_input(
            "Rear Camera (MP)",
            min_value=8,
            max_value=200,
            value=48,
            step=4,
            help="Primary rear camera megapixels"
        )
        
        has_5g = st.selectbox(
            "5G Support",
            ["Yes", "No"],
            index=0,
            help="5G network connectivity support"
        )
        
        has_nfc = st.selectbox(
            "NFC",
            ["Yes", "No"],
            index=0,
            help="Near Field Communication support"
        )
    
    if st.button("🔮 Predict Price Category", type="primary", use_container_width=True):
        # Convert inputs to model format
        refresh_numeric = int(refresh_rate.split()[0])
        
        if fast_charge_power == "No fast charging":
            fc_power = 0
        elif fast_charge_power == "Up to 30W":
            fc_power = 25
        elif fast_charge_power == "31–65W":
            fc_power = 45
        else:
            fc_power = 80
        
        # Build dataframe row
        row = {
            "brand": brand,
            "Processor_Brand": "Qualcomm",
            "Performance_Tier": performance_tier,
            "RAM Size GB": ram_gb,
            "Storage Size GB": storage_gb,
            "Core_Count": 8,
            "Clock_Speed_GHz": 2.4,
            "battery_capacity": battery_capacity,
            "Screen_Size": screen_size,
            "Resolution_Width": 2400,
            "Resolution_Height": 1080,
            "Refresh_Rate": refresh_numeric,
            "fast_charging_power": fc_power,
            "primary_rear_camera_mp": rear_camera_mp,
            "primary_front_camera_mp": 16,
            "num_rear_cameras": 3,
            "num_front_cameras": 1,
            "Dual_Sim": "Yes",
            "4G": "Yes",
            "5G": has_5g,
            "Vo5G": "Yes" if has_5g == "Yes" else "No",
            "NFC": has_nfc,
            "IR_Blaster": "No",
            "memory_card_support": "No",
            "memory_card_size": "No",
            "Notch_Type": "Punch-hole",
            "os_name": "Android",
            "os_version": "13",
            "RAM Tier": "Medium",
            "rating": 85,
        }
        
        return pd.DataFrame([row])
    
    return None


def main():
    """Main application function."""
    # Inject custom CSS
    inject_global_css()
    
    # Initialize sidebar toggle state
    if 'sidebar_visible' not in st.session_state:
        st.session_state.sidebar_visible = True
    
    # Title and header with toggle button
    header_col1, header_col2, header_col3 = st.columns([1, 10, 1])
    
    with header_col1:
        toggle_icon = "☰" if st.session_state.sidebar_visible else "☰"
        toggle_sidebar = st.button(
            toggle_icon,
            help="Toggle Sidebar",
            key="sidebar_toggle",
            use_container_width=False
        )
        if toggle_sidebar:
            st.session_state.sidebar_visible = not st.session_state.sidebar_visible
            st.rerun()
    
    with header_col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 2.5rem; font-weight: 700; margin: 0; padding: 1rem;">
                <span style="font-size: 3rem; margin-right: 0.5rem; display: inline-block;">📱</span>
                <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                Phone Price Classifier
                </span>
            </h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(
        '<p class="subtitle">Predict if a phone is expensive or non-expensive based on specifications</p>',
        unsafe_allow_html=True
    )
    
    # Apply sidebar visibility CSS
    if not st.session_state.sidebar_visible:
        st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            display: none !important;
        }
        [data-testid="stSidebar"] ~ div {
            margin-left: 0 !important;
        }
        [data-testid="stAppViewContainer"] > div:first-child {
            margin-left: 0 !important;
        }
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Load model and artifacts (only if not training)
    model, train_columns, train_df = None, None, None
    
    # Initialize page in session state
    if 'page' not in st.session_state:
        st.session_state.page = "📤 Upload CSV File"
    
    # Sidebar navigation (or top navigation if sidebar is hidden)
    if st.session_state.sidebar_visible:
        # Normal sidebar navigation
        st.sidebar.title("📋 Navigation")
        page = st.sidebar.radio(
            "Choose Option",
            ["📤 Upload CSV File", "✍️ Manual Input", "🤖 Train Model"],
            label_visibility="collapsed",
            index=["📤 Upload CSV File", "✍️ Manual Input", "🤖 Train Model"].index(st.session_state.page) if st.session_state.page in ["📤 Upload CSV File", "✍️ Manual Input", "🤖 Train Model"] else 0
        )
        # Update session state when sidebar selection changes
        if page != st.session_state.page:
            st.session_state.page = page
    else:
        # Top navigation when sidebar is hidden
        st.markdown("### 📋 Navigation")
        nav_col1, nav_col2, nav_col3 = st.columns(3)
        with nav_col1:
            nav_selected = st.button("📤 Upload CSV File", use_container_width=True, key="nav_csv", type="primary" if st.session_state.page == "📤 Upload CSV File" else "secondary")
            if nav_selected:
                st.session_state.page = "📤 Upload CSV File"
                st.rerun()
        with nav_col2:
            nav_selected = st.button("✍️ Manual Input", use_container_width=True, key="nav_manual", type="primary" if st.session_state.page == "✍️ Manual Input" else "secondary")
            if nav_selected:
                st.session_state.page = "✍️ Manual Input"
                st.rerun()
        with nav_col3:
            nav_selected = st.button("🤖 Train Model", use_container_width=True, key="nav_train", type="primary" if st.session_state.page == "🤖 Train Model" else "secondary")
            if nav_selected:
                st.session_state.page = "🤖 Train Model"
                st.rerun()
        
        page = st.session_state.page
    
    # Only load model if not on training page
    if page != "🤖 Train Model":
        with st.spinner("Loading model and artifacts..."):
            model, train_columns, train_df = load_artifacts()
        
        if model is None and page != "🤖 Train Model":
            st.error("""
            ⚠️ **Model not found!**
            
            Please ensure you have:
            1. Trained the model using the "Train Model" page or by running `src/train.py`
            2. The following files exist:
               - `results/final_model.pkl`
               - `results/train_columns.pkl`
               - `data/train.csv`
            """)
            st.stop()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info("""
    This app predicts phone price categories using a trained ensemble model.
    
    **Models Used:**
    - Random Forest
    - Gradient Boosting
    - Neural Network (MLP)
    - CatBoost
    
    Upload a CSV file or manually enter specifications to get predictions.
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    if model is not None:
        st.sidebar.success("✅ Model loaded successfully!")
    else:
        st.sidebar.warning("⚠️ No model loaded")
    
    # Main content area
    if page == "🤖 Train Model":
        st.markdown("## 🤖 Model Training")
        
        st.markdown("""
        <div class="info-box">
        <strong>🚀 Train Your Model:</strong><br>
        This will train an ensemble of 4 machine learning models:<br>
        • Random Forest Classifier<br>
        • Gradient Boosting Classifier<br>
        • Neural Network (MLP)<br>
        • CatBoost Classifier<br><br>
        The models will be combined into an optimized voting ensemble.
        </div>
        """, unsafe_allow_html=True)
        
        # Check if data exists
        if not os.path.exists('../data/train.csv'):
            st.error("❌ Training data not found! Please ensure `data/train.csv` exists.")
            st.stop()
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.info(f"📊 Training data: `data/train.csv`")
        with col2:
            if os.path.exists('../results/final_model.pkl'):
                st.warning("⚠️ Existing model will be overwritten")
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            # Create placeholders for progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()
            
            # Run training
            try:
                training_results = train_model_with_progress(
                    progress_bar,
                    status_text,
                    metrics_placeholder
                )
                
                if training_results:
                    st.markdown("---")
                    st.markdown("### 🎯 Final Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Accuracy",
                            f"{training_results['accuracy']:.4f}"
                        )
                    with col2:
                        st.metric(
                            "Models in Ensemble",
                            training_results['model_count']
                        )
                    with col3:
                        weights_str = str(training_results['weights']) if training_results['model_count'] == 4 else str([w for i, w in enumerate(training_results['weights']) if i != 2])
                        st.metric("Best Weights", weights_str)
                    
                    st.markdown("""
                    <div class="success-box">
                    <strong>✅ Training completed successfully!</strong><br>
                    The model has been saved and is ready to use for predictions.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Reload artifacts
                    model, train_columns, train_df = load_artifacts()
                    
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.exception(e)
    
    elif page == "📤 Upload CSV File":
        if model is None:
            st.error("⚠️ Please train a model first using the 'Train Model' page.")
            st.stop()
        
        st.markdown("## 📤 Batch Prediction (CSV Upload)")
        
        st.markdown("""
        <div class="info-box">
        <strong>📋 Instructions:</strong><br>
        1. Prepare a CSV file with phone specifications<br>
        2. Ensure column names match the training data format<br>
        3. Upload the file below to get batch predictions
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with phone specifications"
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                input_df = pd.read_csv(uploaded_file)
                
                st.markdown("### 📊 Uploaded Data Preview")
                st.dataframe(input_df.head(10), use_container_width=True)
                st.caption(f"Total rows: {len(input_df)}")
                
                # Make predictions
                if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
                    with st.spinner("Processing predictions..."):
                        pred_labels, proba = predict_from_raw_df(
                            input_df,
                            model,
                            train_columns,
                            train_df
                        )
                    
                    if pred_labels is not None:
                        # Create results dataframe
                        result_df = input_df.copy()
                        result_df['Predicted_Price_Category'] = pred_labels
                        result_df['Confidence_Score'] = proba
                        result_df['Confidence_Score'] = result_df['Confidence_Score'].apply(
                            lambda x: f"{x:.2%}"
                        )
                        
                        # Reorder columns: put predictions first
                        other_cols = [col for col in result_df.columns 
                                    if col not in ['Predicted_Price_Category', 'Confidence_Score']]
                        result_df = result_df[['Predicted_Price_Category', 'Confidence_Score'] + other_cols]
                        
                        st.markdown("### ✅ Prediction Results")
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Predictions", value=len(result_df))
                        with col2:
                            expensive_count = sum(1 for p in pred_labels if p == 'expensive')
                            st.metric("Expensive", value=expensive_count)
                        with col3:
                            non_expensive_count = len(pred_labels) - expensive_count
                            st.metric("Non-Expensive", value=non_expensive_count)
                        
                        # Display results table
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Download button
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="phone_price_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        st.markdown("""
                        <div class="success-box">
                        <strong>✅ Predictions completed successfully!</strong><br>
                        You can download the results using the button above.
                        </div>
                        """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your CSV file has the correct format and column names.")
    
    else:  # Manual Input
        if model is None:
            st.error("⚠️ Please train a model first using the 'Train Model' page.")
            st.stop()
        
        st.markdown("## ✍️ Single Prediction (Manual Input)")
        
        st.markdown("""
        <div class="info-box">
        <strong>📝 Instructions:</strong><br>
        Fill in the phone specifications below and click "Predict Price Category" to get
        an instant prediction with confidence score.
        </div>
        """, unsafe_allow_html=True)
        
        # Manual input form
        input_row = manual_input_form()
        
        if input_row is not None:
            with st.spinner("Analyzing phone specifications..."):
                pred_labels, proba = predict_from_raw_df(
                    input_row,
                    model,
                    train_columns,
                    train_df
                )
            
            if pred_labels is not None:
                prediction = pred_labels[0]
                confidence = proba[0]
                
                # Display prediction result
                st.markdown("---")
                st.markdown("### 🎯 Prediction Result")
                
                if prediction == 'expensive':
                    st.markdown(f"""
                    <div class="prediction-card">
                    <h3>💰 Expensive Phone</h3>
                    <p style="font-size: 1.5rem; margin: 0.5rem 0;">
                    Confidence: <strong>{confidence:.1%}</strong>
                    </p>
                    <p style="margin-top: 1rem;">
                    Based on the specifications provided, this phone is predicted to be in the
                    <strong>expensive</strong> category.
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card">
                    <h3>💵 Non-Expensive Phone</h3>
                    <p style="font-size: 1.5rem; margin: 0.5rem 0;">
                    Confidence: <strong>{confidence:.1%}</strong>
                    </p>
                    <p style="margin-top: 1rem;">
                    Based on the specifications provided, this phone is predicted to be in the
                    <strong>non-expensive</strong> category.
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence interpretation
                if confidence >= 0.8:
                    conf_text = "Very High"
                    conf_color = "#4caf50"
                elif confidence >= 0.6:
                    conf_text = "High"
                    conf_color = "#8bc34a"
                elif confidence >= 0.4:
                    conf_text = "Moderate"
                    conf_color = "#ff9800"
                else:
                    conf_text = "Low"
                    conf_color = "#f44336"
                
                st.markdown(f"""
                <div class="prediction-card" style="margin-top: 0.5rem;">
                    <h3>Confidence</h3>
                    <p style="font-size: 1.3rem; margin: 0.25rem 0;">
                        <strong>{conf_text}</strong> ({confidence:.1%})
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 2rem;'>"
        "📱 Phone Price Classifier | Built with Streamlit | "
        "Powered by Ensemble Machine Learning"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
