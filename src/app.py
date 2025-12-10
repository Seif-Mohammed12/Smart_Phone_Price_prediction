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
from pathlib import Path

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
    
    # Title and header
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
    
    # Load model and artifacts
    with st.spinner("Loading model and artifacts..."):
        model, train_columns, train_df = load_artifacts()
    
    if model is None:
        st.error("""
        ⚠️ **Model not found!**
        
        Please ensure you have:
        1. Trained the model by running `src/train.py`
        2. The following files exist:
           - `results/final_model.pkl`
           - `results/train_columns.pkl`
           - `data/train.csv`
        """)
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio(
        "Choose Input Method",
        ["📤 Upload CSV File", "✍️ Manual Input"],
        label_visibility="collapsed"
    )
    
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
    st.sidebar.success("✅ Model loaded successfully!")
    
    # Main content area
    if page == "📤 Upload CSV File":
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
