"""
================================================================================
PHONE PRICE CLASSIFICATION - PREPROCESSING MODULE
================================================================================
This module handles all data preprocessing steps including:
    - K-fold target encoding for categorical features (no data leakage)
    - Feature engineering (creating new informative features)
    - Handling missing values
    - One-hot encoding for categorical variables
    - Column alignment between train and test sets

Key Features:
    - K-fold target encoding prevents data leakage
    - Comprehensive feature engineering for phone specifications
    - Robust handling of missing values
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from utils import memory_card_to_gb


def preprocessing(df, is_train=True, train_df_for_encoding=None):
    """
    Main preprocessing function for phone price classification.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with phone specifications
    is_train : bool, default=True
        Whether this is training data (affects target encoding)
    train_df_for_encoding : pandas.DataFrame, default=None
        Training dataframe used for encoding when is_train=False
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataframe ready for model training/prediction
    """
    df = df.copy()
    
    # ============================================================================
    # SECTION 1: K-FOLD TARGET ENCODING FOR BRAND (NO DATA LEAKAGE!)
    # ============================================================================
    
    if is_train:
        # Convert price labels to numeric
        df['price_target'] = df['price'].map({'non-expensive': 0, 'expensive': 1})
        
        # Use k-fold target encoding to avoid leakage
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        df['brand_expensive_ratio'] = 0.0
        
        # For each fold, use training fold to encode validation fold
        for train_idx, val_idx in kf.split(df):
            train_fold = df.iloc[train_idx]
            val_fold = df.iloc[val_idx]
            
            # Calculate brand means from training fold only
            brand_mean = train_fold.groupby('brand')['price_target'].mean()
            
            # Apply encoding to validation fold
            df.loc[val_idx, 'brand_expensive_ratio'] = df.loc[val_idx, 'brand'].map(brand_mean)
        
        # Fill any remaining NaN with global mean
        global_mean = df['price_target'].mean()
        df['brand_expensive_ratio'] = df['brand_expensive_ratio'].fillna(global_mean)
        
        # Save encoding for test set
        brand_mean_final = df.groupby('brand')['price_target'].mean()
        brand_mean_final.to_pickle('../results/brand_encoding.pkl')
        
    else:
        # For test set, use training data for proper encoding
        if train_df_for_encoding is not None:
            train_df_for_encoding = train_df_for_encoding.copy()
            train_df_for_encoding['price_target'] = train_df_for_encoding['price'].map(
                {'non-expensive': 0, 'expensive': 1}
            )
            brand_mean = train_df_for_encoding.groupby('brand')['price_target'].mean()
            global_mean = train_df_for_encoding['price_target'].mean()
        else:
            # Fallback to saved encoding
            brand_mean = pd.read_pickle('../results/brand_encoding.pkl')
            global_mean = 0.3
            
        df['brand_expensive_ratio'] = df['brand'].map(brand_mean).fillna(global_mean)
    
    # ============================================================================
    # SECTION 2: BINARY COLUMNS CONVERSION
    # ============================================================================
    
    # Convert Yes/No columns to binary (0/1)
    binary_cols = ['Dual_Sim', '4G', '5G', 'Vo5G', 'NFC', 'IR_Blaster']
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna('No').map({'Yes': 1, 'No': 0}).astype(int)
    
    # ============================================================================
    # SECTION 3: FEATURE ENGINEERING
    # ============================================================================
    
    # ----------------------------------------------------------------------------
    # 3.1: Basic Features
    # ----------------------------------------------------------------------------
    
    # Resolution features
    df['resolution_mp'] = df['Resolution_Width'] * df['Resolution_Height'] / 1e6
    df['ppi'] = (
        np.sqrt(df['Resolution_Width']**2 + df['Resolution_Height']**2) /
        df['Screen_Size'].replace(0, np.nan)
    )
    
    # Charging features
    df['has_fast_charging'] = (df['fast_charging_power'] > 0).astype(int)
    df['high_refresh'] = (df['Refresh_Rate'] >= 120).astype(int)
    
    # Memory and storage features
    df['memory_size_gb'] = memory_card_to_gb(df['memory_card_size'])
    df['has_memory_card_slot'] = (
        df['memory_card_support'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    )
    
    # Camera features
    df['total_camera_mp'] = (
        df['primary_rear_camera_mp'].fillna(0) +
        df['primary_front_camera_mp'].fillna(0)
    )
    
    # Processor features
    df['is_flagship_processor'] = df['Performance_Tier'].isin(['Flagship']).astype(int)
    
    # ----------------------------------------------------------------------------
    # 3.2: Screen Quality Features
    # ----------------------------------------------------------------------------
    
    df['screen_quality'] = df['ppi'] * df['Refresh_Rate'] / 1000
    df['screen_area'] = df['Screen_Size'] ** 2
    df['aspect_ratio'] = (
        df['Resolution_Width'] / df['Resolution_Height'].replace(0, np.nan)
    )
    
    # ----------------------------------------------------------------------------
    # 3.3: Premium Feature Combinations
    # ----------------------------------------------------------------------------
    
    # Count of premium features
    df['premium_combo'] = (
        df['5G'] + df['NFC'] + df['has_fast_charging'] +
        df['high_refresh'] + df['is_flagship_processor']
    )
    
    # Luxury score based on screen quality and charging
    df['luxury_score'] = (
        df['ppi'] * df['Refresh_Rate'] * (df['has_fast_charging'] + 1) / 1e6
    )
    
    # Ultra premium feature count
    df['ultra_premium'] = (
        df['is_flagship_processor'] +
        df['5G'] +
        df['NFC'] +
        df['high_refresh'] +
        (df['primary_rear_camera_mp'] >= 64).astype(int) +
        (df['fast_charging_power'] >= 65).astype(int)
    ).clip(0, 6)
    
    # ----------------------------------------------------------------------------
    # 3.4: Brand-Specific Features
    # ----------------------------------------------------------------------------
    
    df['is_apple'] = (
        df['brand'].str.lower().str.contains('apple|iphone', na=False)
    ).astype(int)
    df['is_samsung'] = (
        df['brand'].str.lower().str.contains('samsung', na=False)
    ).astype(int)
    df['is_xiaomi'] = (
        df['brand'].str.lower().str.contains('xiaomi|redmi|poco', na=False)
    ).astype(int)
    
    # ----------------------------------------------------------------------------
    # 3.5: Performance Features
    # ----------------------------------------------------------------------------
    
    df['performance_score'] = (
        df['Core_Count'].fillna(0) * df['Clock_Speed_GHz'].fillna(0)
    )
    df['ram_storage_ratio'] = (
        df['RAM Size GB'] / df['Storage Size GB'].replace(0, np.nan)
    )
    df['battery_per_gram'] = (
        df['battery_capacity'] / df['Screen_Size'].replace(0, np.nan)
    )  # Proxy for efficiency
    
    # ----------------------------------------------------------------------------
    # 3.6: Camera Features
    # ----------------------------------------------------------------------------
    
    df['rear_camera_density'] = (
        df['primary_rear_camera_mp'] / (df['num_rear_cameras'].replace(0, 1))
    )
    df['front_camera_density'] = (
        df['primary_front_camera_mp'] / (df['num_front_cameras'].replace(0, 1))
    )
    df['camera_ratio'] = (
        df['primary_rear_camera_mp'] / (df['primary_front_camera_mp'].replace(0, 1))
    )
    df['high_res_rear'] = (df['primary_rear_camera_mp'] >= 48).astype(int)
    df['multi_rear_cam'] = (df['num_rear_cameras'] >= 3).astype(int)
    
    # ----------------------------------------------------------------------------
    # 3.7: Charging Features
    # ----------------------------------------------------------------------------
    
    df['charging_speed'] = df['fast_charging_power']
    df['super_fast_charging'] = (df['fast_charging_power'] >= 50).astype(int)
    df['ultra_fast_charging'] = (df['fast_charging_power'] >= 65).astype(int)
    
    # ----------------------------------------------------------------------------
    # 3.8: Storage and Memory Tiers
    # ----------------------------------------------------------------------------
    
    # Storage tier: 0=0-64GB, 1=64-128GB, 2=128-256GB, 3=256GB+
    df['storage_tier'] = pd.cut(
        df['Storage Size GB'],
        bins=[0, 64, 128, 256, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(float)
    
    # RAM tier: 0=<4GB, 1=4-8GB, 2=8-12GB, 3=12GB+
    df['ram_tier_numeric'] = df['RAM Size GB'].apply(
        lambda x: 0 if x < 4 else 1 if x < 8 else 2 if x < 12 else 3
    )
    
    # ----------------------------------------------------------------------------
    # 3.9: Interaction Features
    # ----------------------------------------------------------------------------
    
    # Feature interactions
    df['flagship_5g'] = df['is_flagship_processor'] * df['5G']
    df['premium_screen_camera'] = (df['high_refresh'] * df['high_res_rear']).astype(int)
    
    # Total premium features count
    df['total_premium_features'] = (
        df['5G'] + df['NFC'] + df['has_fast_charging'] + df['high_refresh'] +
        df['is_flagship_processor'] + df['high_res_rear'] + df['multi_rear_cam']
    )
    
    # ----------------------------------------------------------------------------
    # 3.10: Rating-Based Features (if available)
    # ----------------------------------------------------------------------------
    
    if 'rating' in df.columns:
        df['high_rating'] = (df['rating'] >= 85).astype(int)
        df['rating_per_price'] = (
            df['rating'] * df['brand_expensive_ratio']
        )  # Interaction feature
    
    # ============================================================================
    # SECTION 4: HANDLE MISSING VALUES
    # ============================================================================
    
    # Fill numeric columns with median (for training) or 0 (for test)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col not in ['price_target', 'price']:
            if is_train:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)
    
    # Fill any remaining NaN with 0
    df = df.fillna(0)
    
    # ============================================================================
    # SECTION 5: ONE-HOT ENCODING
    # ============================================================================
    
    # Categorical columns to one-hot encode
    cat_cols = [
        'Processor_Brand',
        'Performance_Tier',
        'RAM Tier',
        'Notch_Type',
        'os_name'
    ]
    
    # Apply one-hot encoding (drop_first to avoid multicollinearity)
    df = pd.get_dummies(
        df,
        columns=[c for c in cat_cols if c in df.columns],
        drop_first=True
    )
    
    # ============================================================================
    # SECTION 6: FINAL TARGET LABEL PREPARATION
    # ============================================================================
    
    if is_train:
        # Convert price_target to final price column
        df['price'] = df['price_target']
        df = df.drop('price_target', axis=1)
    else:
        # Remove price_target if it exists (shouldn't for test set)
        df = df.drop('price_target', axis=1, errors='ignore')
    
    # ============================================================================
    # SECTION 7: DROP UNNEEDED COLUMNS
    # ============================================================================
    
    # Columns that are redundant or not useful after feature engineering
    drop_cols = [
        'memory_card_size',      # Converted to memory_size_gb
        'memory_card_support',   # Converted to has_memory_card_slot
        'Processor_Series',      # Redundant with Processor_Brand
        'os_version',            # Not informative
        'brand'                  # Encoded as brand_expensive_ratio and brand flags
        # Note: fast_charging_power kept as it's a useful raw feature
    ]
    
    df = df.drop(
        columns=[c for c in drop_cols if c in df.columns],
        errors='ignore'
    )
    
    return df
