"""
Pydantic schemas for structured outputs in Parent Decision Copilot.

This module defines the data models used throughout the application for:
- Intent classification results
- Product recommendations
- Informational responses
- Safety assessments
- API request/response models
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class IntentType(str, Enum):
    """Enumeration of supported user intent types."""
    PRODUCT_RECOMMENDATION = "product_recommendation"
    INFORMATIONAL_QUERY = "informational_query"
    COMPARISON_QUERY = "comparison_query"
    HEALTH_SENSITIVE_QUERY = "health_sensitive_query"


class SafetyLevel(str, Enum):
    """Safety classification levels for user queries."""
    SAFE = "safe"
    CAUTION = "caution"
    DANGEROUS = "dangerous"


class ConfidenceScore(BaseModel):
    """Confidence scoring for model predictions."""
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: Optional[str] = Field(None, description="Explanation for the confidence score")


class ProductInfo(BaseModel):
    """Product information structure."""
    id: str = Field(..., description="Unique product identifier")
    name: str = Field(..., description="Product name")
    category: str = Field(..., description="Product category")
    price_range: Optional[str] = Field(None, description="Price range indicator")
    features: List[str] = Field(default_factory=list, description="Key product features")
    safety_rating: Optional[float] = Field(None, ge=1.0, le=5.0, description="Safety rating (1-5)")
    recommended_age: Optional[str] = Field(None, description="Recommended age range")


class KnowledgeInfo(BaseModel):
    """Parenting knowledge information structure."""
    id: str = Field(..., description="Unique knowledge entry identifier")
    title: str = Field(..., description="Knowledge entry title")
    category: str = Field(..., description="Knowledge category")
    content: str = Field(..., description="Knowledge content")
    source: Optional[str] = Field(None, description="Information source")
    age_relevance: Optional[str] = Field(None, description="Relevant age range")


class IntentClassification(BaseModel):
    """Intent classification result."""
    intent: IntentType = Field(..., description="Classified user intent")
    confidence: ConfidenceScore = Field(..., description="Confidence in classification")
    safety_level: SafetyLevel = Field(..., description="Safety assessment level")


class RecommendationResult(BaseModel):
    """Product recommendation result."""
    products: List[ProductInfo] = Field(..., description="Recommended products")
    reasoning: str = Field(..., description="Explanation for recommendations")
    confidence: ConfidenceScore = Field(..., description="Confidence in recommendations")
    safety_note: Optional[str] = Field(None, description="Safety-related notes")


class InformationalResult(BaseModel):
    """Informational query result."""
    knowledge: List[KnowledgeInfo] = Field(..., description="Relevant knowledge entries")
    summary: str = Field(..., description="Summary of findings")
    confidence: ConfidenceScore = Field(..., description="Confidence in response")
    safety_note: Optional[str] = Field(None, description="Safety-related notes")


class ComparisonResult(BaseModel):
    """Product comparison result."""
    products: List[ProductInfo] = Field(..., description="Products being compared")
    comparison_matrix: Dict[str, Any] = Field(..., description="Comparison details")
    recommendation: Optional[str] = Field(None, description="Comparison-based recommendation")
    confidence: ConfidenceScore = Field(..., description="Confidence in comparison")
    safety_note: Optional[str] = Field(None, description="Safety-related notes")


class SafetyAssessment(BaseModel):
    """Safety assessment for user queries."""
    is_safe: bool = Field(..., description="Whether the query is safe to answer")
    safety_level: SafetyLevel = Field(..., description="Safety classification")
    refusal_reason: Optional[str] = Field(None, description="Reason for refusal if unsafe")
    recommended_action: Optional[str] = Field(None, description="Recommended action for user")


class UserQuery(BaseModel):
    """User query input model."""
    query: str = Field(..., min_length=1, max_length=1000, description="User's query text")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context information")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")

    @validator('query')
    def validate_query(cls, v):
        """Validate query text."""
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()


class ApiResponse(BaseModel):
    """Standard API response model."""
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if failed")
    intent_classification: Optional[IntentClassification] = Field(None, description="Query intent classification")
    safety_assessment: Optional[SafetyAssessment] = Field(None, description="Query safety assessment")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class HealthSensitiveResponse(BaseModel):
    """Response for health-sensitive queries."""
    refusal_message: str = Field(..., description="Professional consultation refusal message")
    recommended_professional: str = Field(..., description="Recommended professional type")
    urgency_level: Optional[str] = Field(None, description="Urgency level indicator")
    general_guidance: Optional[str] = Field(None, description="General non-medical guidance")