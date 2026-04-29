"""
Main FastAPI application for Parent Decision Copilot.

This module provides the REST API endpoints for the AI assistant,
including intent classification, safety assessment, and response generation.
"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    UserQuery, ApiResponse, IntentType, SafetyLevel,
    RecommendationResult, InformationalResult, ComparisonResult,
    HealthSensitiveResponse, ConfidenceScore, ProductInfo, KnowledgeInfo
)
from .classifier import create_classifier
from .safety import create_safety_manager
from .retrieval import create_retrieval_system
from .recommender import create_recommendation_engine


# Initialize FastAPI application
app = FastAPI(
    title="Parent Decision Copilot API",
    description="A safety-aware AI assistant for parenting and baby-product decision support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline components
classifier = create_classifier()
safety_manager = create_safety_manager()
retrieval_system = create_retrieval_system()
recommendation_engine = create_recommendation_engine()

# Data file paths
DATA_DIR = Path(__file__).parent / "data"
PRODUCTS_FILE = DATA_DIR / "products.json"
KNOWLEDGE_FILE = DATA_DIR / "parenting_knowledge.json"

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# In-memory data storage (loaded from JSON files)
products_data: Dict[str, Any] = {}
knowledge_data: Dict[str, Any] = {}


def load_data_files():
    """Load product and knowledge data from JSON files."""
    global products_data, knowledge_data
    
    try:
        if PRODUCTS_FILE.exists():
            with open(PRODUCTS_FILE, 'r', encoding='utf-8') as f:
                products_data = json.load(f)
        else:
            products_data = {"products": []}
            print(f"Warning: Products file not found at {PRODUCTS_FILE}")
        
        if KNOWLEDGE_FILE.exists():
            with open(KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
        else:
            knowledge_data = {"knowledge": []}
            print(f"Warning: Knowledge file not found at {KNOWLEDGE_FILE}")
            
    except Exception as e:
        print(f"Error loading data files: {e}")
        products_data = {"products": []}
        knowledge_data = {"knowledge": []}


# Load data on startup
@app.on_event("startup")
def startup_event():
    """Initialize application on startup."""
    load_data_files()


@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Parent Decision Copilot API",
        "version": "1.0.0",
        "description": "Safety-aware AI assistant for parenting decisions",
        "endpoints": {
            "analyze": "/analyze - Complete pipeline analysis",
            "classify": "/classify - Classify user intent",
            "health": "/health - Health check endpoint"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        products_count = len(retrieval_system.products_data)
        knowledge_count = len(retrieval_system.knowledge_data)
        
        return {
            "status": "healthy",
            "data_loaded": products_count > 0 and knowledge_count > 0,
            "products_count": products_count,
            "knowledge_count": knowledge_count,
            "components": {
                "classifier": "ready",
                "safety_manager": "ready", 
                "retrieval_system": "ready",
                "recommendation_engine": "ready"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/classify")
async def classify_intent(user_query: UserQuery) -> ApiResponse:
    """
    Classify user intent without generating a full response.
    
    This endpoint performs intent classification and safety assessment
    but does not generate recommendations or retrieve data.
    """
    try:
        logger.info(f"Classifying intent for query: {user_query.query[:100]}...")
        
        # Classify intent
        intent_result = classifier.classify_intent(user_query.query)
        
        # Perform safety assessment
        safety_assessment = safety_manager.assess_safety(user_query.query, intent_result.intent)
        
        logger.info(f"Intent classified: {intent_result.intent.value}, Safety: {safety_assessment.safety_level.value}")
        
        return ApiResponse(
            success=True,
            data={
                "intent": intent_result.intent.value,
                "intent_explanation": classifier.get_intent_explanation(intent_result.intent),
                "confidence": intent_result.confidence.dict(),
                "safety_level": intent_result.safety_level.value
            },
            intent_classification=intent_result,
            safety_assessment=safety_assessment,
            metadata={
                "query_length": len(user_query.query),
                "processing_type": "classification_only"
            }
        )
        
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return ApiResponse(
            success=False,
            error=f"Classification failed: {str(e)}",
            metadata={"error_type": "classification_error"}
        )


@app.post("/analyze")
async def analyze_query(user_query: UserQuery) -> ApiResponse:
    """
    Complete pipeline analysis of user query.
    
    This endpoint performs the full end-to-end workflow:
    1. Intent classification
    2. Safety assessment
    3. Retrieval (if safe)
    4. Recommendation generation (if safe)
    5. Structured response
    """
    try:
        logger.info(f"Starting analysis for query: {user_query.query[:100]}...")
        
        # Step 1: Classify intent
        intent_result = classifier.classify_intent(user_query.query)
        logger.info(f"Intent: {intent_result.intent.value} (confidence: {intent_result.confidence.score:.2f})")
        
        # Step 2: Safety assessment
        safety_assessment = safety_manager.assess_safety(user_query.query, intent_result.intent)
        logger.info(f"Safety level: {safety_assessment.safety_level.value}")
        
        # Step 3: Check if safe to proceed
        if not safety_manager.is_safe_to_proceed(safety_assessment):
            logger.info("Query not safe to proceed - returning health-sensitive response")
            
            # Handle health-sensitive or dangerous queries
            health_response = safety_manager.create_health_sensitive_response(
                user_query.query, 
                safety_assessment.safety_level
            )
            
            return ApiResponse(
                success=True,
                data=health_response.dict(),
                intent_classification=intent_result,
                safety_assessment=safety_assessment,
                metadata={
                    "response_type": "health_sensitive_refusal",
                    "safety_action": "refused_medical_advice",
                    "pipeline_steps": ["classification", "safety_assessment", "refusal"]
                }
            )
        
        # Step 4: Process based on intent type
        response_data = None
        response_type = None
        sources = []
        
        if intent_result.intent == IntentType.PRODUCT_RECOMMENDATION:
            logger.info("Processing product recommendation")
            response_data = recommendation_engine.generate_product_recommendations(
                user_query.query, 
                intent_result.confidence.score
            )
            response_type = "product_recommendation"
            sources = ["local_products_database"]
            
        elif intent_result.intent == IntentType.INFORMATIONAL_QUERY:
            logger.info("Processing informational query")
            response_data = recommendation_engine.generate_informational_response(
                user_query.query,
                intent_result.confidence.score
            )
            response_type = "informational_response"
            sources = ["local_knowledge_database"]
            
        elif intent_result.intent == IntentType.COMPARISON_QUERY:
            logger.info("Processing comparison query")
            response_data = recommendation_engine.generate_comparison_response(
                user_query.query,
                intent_result.confidence.score
            )
            response_type = "comparison_response"
            sources = ["local_products_database"]
            
        else:
            # Fallback for any unexpected cases
            logger.warning(f"Unexpected intent type: {intent_result.intent.value}")
            response_data = {
                "message": "I'm here to help with parenting decisions, but I cannot provide medical advice. For health concerns, please consult a healthcare professional.",
                "safety_note": safety_manager.get_safety_disclaimer()
            }
            response_type = "general_safety_response"
            sources = []
        
        # Add uncertainty messaging if confidence is low
        uncertainty_note = None
        if hasattr(response_data, 'confidence') and response_data.confidence.score < 0.3:
            uncertainty_note = "Low confidence results - consider expanding search criteria or consulting professionals."
        elif isinstance(response_data, dict) and 'confidence' in response_data:
            if response_data['confidence'].get('score', 1.0) < 0.3:
                uncertainty_note = "Low confidence results - consider expanding search criteria or consulting professionals."
        
        logger.info(f"Analysis complete. Response type: {response_type}")
        
        return ApiResponse(
            success=True,
            data=response_data.dict() if hasattr(response_data, 'dict') else response_data,
            intent_classification=intent_result,
            safety_assessment=safety_assessment,
            metadata={
                "response_type": response_type,
                "sources": sources,
                "uncertainty_note": uncertainty_note,
                "pipeline_steps": ["classification", "safety_assessment", "retrieval", "recommendation"]
            }
        )
        
    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        return ApiResponse(
            success=False,
            error=f"Query processing failed: {str(e)}",
            metadata={
                "error_type": "processing_error",
                "pipeline_steps": ["classification", "safety_assessment", "error"]
            }
        )


@app.post("/query")
async def process_query(user_query: UserQuery) -> ApiResponse:
    """
    Legacy endpoint for backward compatibility.
    
    Redirects to /analyze for the complete pipeline.
    """
    logger.info("Legacy /query endpoint called - redirecting to /analyze")
    return await analyze_query(user_query)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ApiResponse(
            success=False,
            error="An unexpected error occurred. Please try again later.",
            metadata={"error_type": "server_error"}
        ).dict()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception handler."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ApiResponse(
            success=False,
            error=exc.detail,
            metadata={"error_type": "http_error", "status_code": exc.status_code}
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Parent Decision Copilot API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)