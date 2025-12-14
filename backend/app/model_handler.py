import joblib
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

class ModelHandler:
    """Handles loading and prediction for all ML models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all trained models from disk"""
        try:
            # Load models
            lr_path = self.models_dir / 'logistic_regression_model.pkl'
            rf_path = self.models_dir / 'random_forest_model.pkl'
            xgb_path = self.models_dir / 'xgboost_model.pkl'
            
            if lr_path.exists():
                self.models['logistic'] = joblib.load(lr_path)
                print("✅ Logistic Regression model loaded")
            
            if rf_path.exists():
                self.models['random_forest'] = joblib.load(rf_path)
                print("✅ Random Forest model loaded")
            
            if xgb_path.exists():
                self.models['xgboost'] = joblib.load(xgb_path)
                print("✅ XGBoost model loaded")
            
            if not self.models:
                raise Exception("No models found! Please add .pkl files to models/ folder")
            
            print(f"✅ Successfully loaded {len(self.models)} models")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def predict(self, input_df: pd.DataFrame, model_name: str = 'xgboost') -> Tuple[int, float]:
        """
        Make prediction using specified model
        
        Args:
            input_df: Preprocessed input DataFrame
            model_name: Name of model to use
        
        Returns:
            Tuple of (prediction, probability)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probability of class 1 (purchase)
        
        return int(prediction), float(probability)
    
    def predict_all_models(self, input_df: pd.DataFrame) -> Dict:
        """Get predictions from all models"""
        results = {}
        
        for model_name in self.models.keys():
            prediction, probability = self.predict(input_df, model_name)
            results[model_name] = {
                'prediction': prediction,
                'probability': probability
            }
        
        return results
    
    def get_ensemble_prediction(self, input_df: pd.DataFrame) -> Tuple[int, float]:
        """
        Get ensemble prediction by averaging probabilities from all models
        
        Returns:
            Tuple of (prediction, average_probability)
        """
        all_predictions = self.predict_all_models(input_df)
        
        # Average probabilities
        avg_probability = np.mean([pred['probability'] for pred in all_predictions.values()])
        
        # Threshold at 0.5
        ensemble_prediction = 1 if avg_probability >= 0.5 else 0
        
        return ensemble_prediction, float(avg_probability)
    
    def get_model_performance(self) -> Dict:
        """Return stored model performance metrics from your training"""
        return {
            'logistic': {
                'accuracy': 0.868,
                'precision': 0.552,
                'recall': 0.811,
                'f1_score': 0.657,
                'roc_auc': 0.913
            },
            'random_forest': {
                'accuracy': 0.883,
                'precision': 0.636,
                'recall': 0.589,
                'f1_score': 0.612,
                'roc_auc': 0.882
            },
            'xgboost': {
                'accuracy': 0.865,
                'precision': 0.549,
                'recall': 0.776,
                'f1_score': 0.643,
                'roc_auc': 0.895
            }
        }
    
    def get_feature_importance(self, model_name: str = 'xgboost') -> List[Dict]:
        """Get feature importance from tree-based model"""
        
        if model_name not in ['random_forest', 'xgboost']:
            return []
        
        if model_name not in self.models:
            return []
        
        model = self.models[model_name]
        
        # Feature names (reduced set)
        feature_names = [
            'has_value_page',
            'value_per_page', 
            'PageValues',
            'interaction_score',
            'ProductRelated',
            'Month_Nov'
        ]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create list of feature importance dicts
            importance_list = [
                {'feature': name, 'importance': float(imp)}
                for name, imp in zip(feature_names, importances)
            ]
            
            # Sort by importance
            importance_list.sort(key=lambda x: x['importance'], reverse=True)
            
            return importance_list
        
        return []
    
    @staticmethod
    def get_confidence_level(probability: float) -> str:
        """Determine confidence level based on probability"""
        if probability < 0.3 or probability > 0.7:
            return "High"
        elif probability < 0.4 or probability > 0.6:
            return "Medium"
        else:
            return "Low"
    
    @staticmethod
    def get_recommendation(prediction: int, probability: float, input_data: Dict) -> str:
        """Generate actionable recommendation based on prediction"""
        
        if prediction == 1 and probability > 0.7:
            return "High purchase intent detected! Consider offering a limited-time discount or free shipping to seal the deal."
        
        elif prediction == 1 and probability > 0.5:
            return "Moderate purchase intent. Send personalized product recommendations or show customer reviews to build confidence."
        
        elif prediction == 0 and probability < 0.3:
            return "Low engagement detected. Consider retargeting with dynamic ads showcasing products they viewed."
        
        else:
            # Check specific issues
            if input_data.get('BounceRates', 0) > 0.1:
                return "High bounce rate detected. Improve page load speed and ensure clear navigation paths."
            elif input_data.get('ProductRelated', 0) < 5:
                return "Limited product exploration. Show related products and bestsellers to increase engagement."
            else:
                return "Borderline case. Use exit-intent popups with incentives to capture email and retarget later."