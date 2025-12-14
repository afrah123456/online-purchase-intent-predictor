
from pydantic import BaseModel, Field
from typing import Optional, List

class SessionInput(BaseModel):
    """Input schema for a single session prediction"""
    Administrative: int = Field(ge=0, description="Number of administrative pages visited")
    Informational: int = Field(ge=0, description="Number of informational pages visited")
    ProductRelated: int = Field(ge=0, description="Number of product-related pages visited")
    BounceRates: float = Field(ge=0, le=1, description="Bounce rate of the session")
    PageValues: float = Field(ge=0, description="Average page value")
    SpecialDay: float = Field(ge=0, le=1, description="Closeness to special day (0-1)")
    Month: str = Field(description="Month of visit")
    OperatingSystems: int = Field(ge=1, description="Operating system ID")
    Browser: int = Field(ge=1, description="Browser ID")
    Region: int = Field(ge=1, description="Region ID")
    TrafficType: int = Field(ge=1, description="Traffic type ID")
    VisitorType: str = Field(description="Type of visitor")
    Weekend: bool = Field(description="Whether the session occurred on weekend")

    class Config:
        json_schema_extra = {
            "example": {
                "Administrative": 2,
                "Informational": 0,
                "ProductRelated": 15,
                "BounceRates": 0.02,
                "PageValues": 25.5,
                "SpecialDay": 0.0,
                "Month": "Nov",
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 2,
                "VisitorType": "Returning_Visitor",
                "Weekend": False
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    prediction: int = Field(description="0 = Won't Purchase, 1 = Will Purchase")
    probability: float = Field(description="Probability of purchase (0-1)")
    confidence: str = Field(description="Confidence level (Low/Medium/High)")
    model_used: str = Field(description="Model used for prediction")
    top_features: List[dict] = Field(description="Top influencing features")
    recommendation: str = Field(description="Actionable recommendation")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    total_sessions: int
    predicted_purchases: int
    predicted_abandons: int
    avg_purchase_probability: float
    predictions: List[dict]


class ModelPerformance(BaseModel):
    """Model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float


class FeatureImportance(BaseModel):
    """Feature importance for a model"""
    feature: str
    importance: float