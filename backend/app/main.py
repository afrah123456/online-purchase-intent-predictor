from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
import io
import time
from datetime import datetime

from schemas import (
    SessionInput,
    PredictionResponse,
    BatchPredictionResponse,
    ModelPerformance,
    FeatureImportance
)
from model_handler import ModelHandler
from preprocessing import preprocess_input, preprocess_batch
from risk_scorer import RiskScorer

# Initialize FastAPI app
app = FastAPI(
    title="Purchase Intent Prediction API",
    description="ML-powered API for predicting online purchase intent with real-time risk scoring",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model handler and risk scorer
model_handler = ModelHandler(models_dir="models")
risk_scorer = RiskScorer(model_handler)

# Store API metrics in memory (in production, use database)
api_metrics = {
    'total_requests': 0,
    'total_errors': 0,
    'avg_response_time': 0,
    'requests_by_model': {},
    'recent_requests': []
}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Purchase Intent Prediction API with Real-Time Risk Scoring",
        "status": "active",
        "version": "2.0.0",
        "available_models": list(model_handler.models.keys()),
        "endpoints": {
            "predict": "/predict",
            "risk_score": "/api/risk-score",
            "batch_predict": "/batch-predict",
            "model_performance": "/model-performance",
            "feature_importance": "/feature-importance",
            "api_metrics": "/api/metrics"
        }
    }


@app.post("/api/risk-score")
async def get_risk_score(
        session: SessionInput,
        model_name: str = "xgboost"
):
    """
    Real-time risk scoring endpoint for production use

    Fast risk assessment with actionable recommendations
    Designed for integration with live e-commerce systems

    Args:
        session: Customer session data
        model_name: Model to use (default: xgboost for best recall)

    Returns:
        Comprehensive risk assessment with actions
    """
    start_time = time.time()

    try:
        # Update metrics
        api_metrics['total_requests'] += 1
        api_metrics['requests_by_model'][model_name] = api_metrics['requests_by_model'].get(model_name, 0) + 1

        # Convert input to dict
        input_data = session.model_dump()

        # Preprocess input
        processed_df = preprocess_input(input_data, use_reduced=True)

        # Make prediction
        prediction, probability = model_handler.predict(processed_df, model_name)

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000

        # Generate comprehensive risk report
        risk_report = risk_scorer.generate_risk_report(
            session_data=input_data,
            prediction=prediction,
            probability=probability,
            model_name=model_name,
            response_time_ms=response_time_ms
        )

        # Update average response time
        total_time = api_metrics['avg_response_time'] * (api_metrics['total_requests'] - 1)
        api_metrics['avg_response_time'] = (total_time + response_time_ms) / api_metrics['total_requests']

        # Store recent request (keep last 100)
        api_metrics['recent_requests'].append({
            'timestamp': risk_report['timestamp'],
            'risk_level': risk_report['risk_assessment']['risk_level'],
            'response_time_ms': response_time_ms,
            'model': model_name
        })
        if len(api_metrics['recent_requests']) > 100:
            api_metrics['recent_requests'].pop(0)

        return risk_report

    except Exception as e:
        api_metrics['total_errors'] += 1
        raise HTTPException(status_code=500, detail=f"Risk scoring error: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_purchase_intent(
        session: SessionInput,
        model_name: str = "xgboost"
):
    """
    Predict purchase intent for a single session

    Args:
        session: Session data
        model_name: Model to use (logistic, random_forest, xgboost, ensemble)

    Returns:
        Prediction with probability and recommendations
    """
    start_time = time.time()

    try:
        # Update metrics
        api_metrics['total_requests'] += 1
        api_metrics['requests_by_model'][model_name] = api_metrics['requests_by_model'].get(model_name, 0) + 1

        # Convert input to dict
        input_data = session.model_dump()

        # Preprocess input
        processed_df = preprocess_input(input_data, use_reduced=True)

        # Make prediction
        if model_name == "ensemble":
            prediction, probability = model_handler.get_ensemble_prediction(processed_df)
            model_used = "Ensemble (All Models)"
        else:
            prediction, probability = model_handler.predict(processed_df, model_name)
            model_used = model_name.replace('_', ' ').title()

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000

        # Get confidence level
        confidence = model_handler.get_confidence_level(probability)

        # Get top features
        top_features = model_handler.get_feature_importance(
            'xgboost' if model_name == 'ensemble' else model_name
        )[:3]

        # Get recommendation
        recommendation = model_handler.get_recommendation(prediction, probability, input_data)

        # Calculate risk level for metrics
        if prediction == 0:
            risk_score = probability
        else:
            risk_score = 1 - probability

        if risk_score >= 0.80:
            risk_level = 'CRITICAL'
        elif risk_score >= 0.60:
            risk_level = 'HIGH'
        elif risk_score >= 0.40:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        # Update average response time
        total_time = api_metrics['avg_response_time'] * (api_metrics['total_requests'] - 1)
        api_metrics['avg_response_time'] = (total_time + response_time_ms) / api_metrics['total_requests']

        # Store recent request
        api_metrics['recent_requests'].append({
            'timestamp': datetime.utcnow().isoformat(),
            'risk_level': risk_level,
            'response_time_ms': response_time_ms,
            'model': model_name
        })
        if len(api_metrics['recent_requests']) > 100:
            api_metrics['recent_requests'].pop(0)

        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            confidence=confidence,
            model_used=model_used,
            top_features=top_features,
            recommendation=recommendation
        )

    except Exception as e:
        api_metrics['total_errors'] += 1
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(
        file: UploadFile = File(...),
        model_name: str = "xgboost"
):
    """
    Batch prediction from CSV file

    Args:
        file: CSV file with session data
        model_name: Model to use for predictions

    Returns:
        Batch prediction results
    """
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Validate required columns
        required_cols = ['Administrative', 'Informational', 'ProductRelated',
                         'BounceRates', 'PageValues', 'SpecialDay', 'Month',
                         'OperatingSystems', 'Browser', 'Region', 'TrafficType',
                         'VisitorType', 'Weekend']

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )

        # Preprocess batch
        processed_df = preprocess_batch(df.copy(), use_reduced=True)

        # Make predictions
        predictions = []
        probabilities = []

        for idx in range(len(processed_df)):
            single_row = processed_df.iloc[idx:idx + 1]
            pred, prob = model_handler.predict(single_row, model_name)
            predictions.append(pred)
            probabilities.append(prob)

        # Create results
        results = []
        for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'session_id': idx + 1,
                'prediction': int(pred),
                'probability': round(float(prob), 4),
                'confidence': model_handler.get_confidence_level(prob)
            })

        # Calculate statistics
        predicted_purchases = sum(predictions)
        predicted_abandons = len(predictions) - predicted_purchases
        avg_probability = sum(probabilities) / len(probabilities)

        return BatchPredictionResponse(
            total_sessions=len(predictions),
            predicted_purchases=predicted_purchases,
            predicted_abandons=predicted_abandons,
            avg_purchase_probability=round(avg_probability, 4),
            predictions=results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model-performance", response_model=List[ModelPerformance])
async def get_model_performance():
    """Get performance metrics for all models"""
    try:
        performance = model_handler.get_model_performance()

        results = []
        for model_name, metrics in performance.items():
            results.append(ModelPerformance(
                model_name=model_name.replace('_', ' ').title(),
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                roc_auc=metrics['roc_auc']
            ))

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/feature-importance", response_model=List[FeatureImportance])
async def get_feature_importance(model_name: str = "xgboost"):
    """Get feature importance for specified model"""
    try:
        if model_name not in ['random_forest', 'xgboost']:
            raise HTTPException(
                status_code=400,
                detail="Feature importance only available for random_forest and xgboost"
            )

        importance = model_handler.get_feature_importance(model_name)

        return [
            FeatureImportance(feature=item['feature'], importance=item['importance'])
            for item in importance
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/metrics")
async def get_api_metrics():
    """Get API performance metrics for monitoring dashboard"""

    # Calculate risk level distribution
    risk_distribution = {
        'CRITICAL': 0,
        'HIGH': 0,
        'MEDIUM': 0,
        'LOW': 0
    }

    for req in api_metrics['recent_requests']:
        risk_level = req.get('risk_level', 'MEDIUM')
        risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1

    return {
        'total_requests': api_metrics['total_requests'],
        'total_errors': api_metrics['total_errors'],
        'error_rate': round((api_metrics['total_errors'] / max(api_metrics['total_requests'], 1)) * 100, 2),
        'avg_response_time_ms': round(api_metrics['avg_response_time'], 2),
        'requests_by_model': api_metrics['requests_by_model'],
        'risk_distribution': risk_distribution,
        'recent_requests': api_metrics['recent_requests'][-20:],  # Last 20 requests
        'uptime': 'operational',
        'models_loaded': len(model_handler.models)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(model_handler.models),
        "available_models": list(model_handler.models.keys()),
        "total_api_calls": api_metrics['total_requests']
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)