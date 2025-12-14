import pandas as pd
import numpy as np
from typing import Dict

def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all engineered features from your ML project"""
    
    # Total pages visited
    df['total_pages_visited'] = (
        df['Administrative'] + df['Informational'] + df['ProductRelated']
    )
    
    # Binary flags for page views
    df['has_product_views'] = (df['ProductRelated'] > 0).astype(int)
    df['has_info_views'] = (df['Informational'] > 0).astype(int)
    df['has_admin_views'] = (df['Administrative'] > 0).astype(int)
    
    # Admin-only session flag
    df['admin_only_session'] = (
        (df['Administrative'] > 0) &
        (df['Informational'] == 0) &
        (df['ProductRelated'] == 0)
    ).astype(int)
    
    # Value per page (avoid division by zero)
    eps = 1.0
    df['value_per_page'] = df['PageValues'] / (df['total_pages_visited'] + eps)
    
    # Interaction score
    df['interaction_score'] = df['PageValues'] - df['BounceRates'] * df['total_pages_visited']
    
    # Has value page flag
    df['has_value_page'] = (df['PageValues'] > 0).astype(int)
    
    # Special weekend flag
    df['special_weekend'] = (df['SpecialDay'] > 0).astype(int) * df['Weekend'].astype(int)
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features using one-hot encoding"""
    
    # Convert Weekend to int if it's boolean
    if df['Weekend'].dtype == bool:
        df['Weekend'] = df['Weekend'].astype(int)
    
    # One-hot encode categorical columns
    df = pd.get_dummies(
        df,
        columns=['Month', 'VisitorType', 'OperatingSystems', 'Browser', 'Region', 'TrafficType'],
        drop_first=True
    )
    
    # Add is_returning_customer feature
    df['is_returning_customer'] = df.get('VisitorType_Returning_Visitor', 0)
    
    return df


def get_reduced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select only the 6 core features used in reduced models"""
    
    core_features = [
        'has_value_page',
        'value_per_page', 
        'PageValues',
        'interaction_score',
        'ProductRelated',
        'Month_Nov'
    ]
    
    # Ensure all features exist, add missing ones as 0
    for feature in core_features:
        if feature not in df.columns:
            df[feature] = 0
    
    return df[core_features]


def preprocess_input(input_data: Dict, use_reduced: bool = True) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for a single input
    
    Args:
        input_data: Dictionary containing session data
        use_reduced: Whether to return only reduced feature set
    
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    # Create engineered features
    df = create_engineered_features(df)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Get reduced features if specified
    if use_reduced:
        df = get_reduced_features(df)
    
    return df


def preprocess_batch(df: pd.DataFrame, use_reduced: bool = True) -> pd.DataFrame:
    """
    Preprocess a batch of sessions from CSV upload
    
    Args:
        df: DataFrame containing multiple sessions
        use_reduced: Whether to return only reduced feature set
    
    Returns:
        Preprocessed DataFrame ready for batch prediction
    """
    
    # Create engineered features
    df = create_engineered_features(df)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Get reduced features if specified
    if use_reduced:
        df = get_reduced_features(df)
    
    return df