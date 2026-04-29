"""
Main FastAPI application for Parent Decision Copilot.

This module provides the REST API endpoints for the AI assistant,
including intent classification, safety assessment, and response generation.
"""

import json
import os
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

# Initialize components
classifier = create_classifier()
safety_manager = create_safety_manager()

# Data file paths
DATA_DIR = Path(__file__).parent / "data"
PRODUCTS_FILE = DATA_DIR / "products.json"
KNOWLEDGE_FILE = DATA_DIR / "parenting_knowledge.json"

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
            "classify": "/classify - Classify user intent",
            "query": "/query - Process user query with safety assessment",
            "health": "/health - Health check endpoint"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "data_loaded": len(products_data.get("products", [])) > 0 and len(knowledge_data.get("knowledge", [])) > 0,
        "products_count": len(products_data.get("products", [])),
        "knowledge_count": len(knowledge_data.get("knowledge", []))
    }


@app.post("/classify")
async def classify_intent(user_query: UserQuery) -> ApiResponse:
    """
    Classify user intent without generating a full response.
    
    This endpoint performs intent classification and safety assessment
    but does not generate recommendations or retrieve data.
    """
    try:
        # Classify intent
        intent_result = classifier.classify_intent(user_query.query)
        
        # Perform safety assessment
        safety_assessment = safety_manager.assess_safety(user_query.query, intent_result.intent)
        
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
        return ApiResponse(
            success=False,
            error=f"Classification failed: {str(e)}",
            metadata={"error_type": "classification_error"}
        )


@app.post("/query")
async def process_query(user_query: UserQuery) -> ApiResponse:
    """
    Process user query with full response generation.
    
    This endpoint performs intent classification, safety assessment,
    and generates appropriate responses based on the intent type.
    """
    try:
        # Classify intent
        intent_result = classifier.classify_intent(user_query.query)
        
        # Perform safety assessment
        safety_assessment = safety_manager.assess_safety(user_query.query, intent_result.intent)
        
        # Check if it's safe to proceed
        if not safety_manager.is_safe_to_proceed(safety_assessment):
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
                    "safety_action": "refused_medical_advice"
                }
            )
        
        # Process based on intent type
        response_data = None
        response_type = None
        
        if intent_result.intent == IntentType.PRODUCT_RECOMMENDATION:
            response_data, response_type = await handle_product_recommendation(user_query, intent_result)
            
        elif intent_result.intent == IntentType.INFORMATIONAL_QUERY:
            response_data, response_type = await handle_informational_query(user_query, intent_result)
            
        elif intent_result.intent == IntentType.COMPARISON_QUERY:
            response_data, response_type = await handle_comparison_query(user_query, intent_result)
            
        else:
            # Fallback for any unexpected cases
            response_data = {
                "message": "I'm here to help with parenting decisions, but I cannot provide medical advice. For health concerns, please consult a healthcare professional.",
                "safety_note": safety_manager.get_safety_disclaimer()
            }
            response_type = "general_safety_response"
        
        return ApiResponse(
            success=True,
            data=response_data,
            intent_classification=intent_result,
            safety_assessment=safety_assessment,
            metadata={
                "response_type": response_type,
                "data_sources": ["local_json_files"]
            }
        )
        
    except Exception as e:
        return ApiResponse(
            success=False,
            error=f"Query processing failed: {str(e)}",
            metadata={"error_type": "processing_error"}
        )


async def handle_product_recommendation(user_query: UserQuery, intent_result) -> tuple[Dict[str, Any], str]:
    """Handle product recommendation queries."""
    try:
        # Simple keyword-based product search (can be enhanced)
        query_lower = user_query.query.lower()
        all_products = products_data.get("products", [])
        
        # Filter products based on keywords
        matching_products = []
        for product in all_products:
            product_text = f"{product.get('name', '')} {product.get('category', '')} {' '.join(product.get('features', []))}".lower()
            if any(keyword in product_text for keyword in ['baby', 'infant', 'toddler', 'child']):
                matching_products.append(product)
        
        # Convert to ProductInfo objects
        product_infos = []
        for product in matching_products[:5]:  # Limit to top 5
            product_infos.append(ProductInfo(
                id=product.get('id', ''),
                name=product.get('name', ''),
                category=product.get('category', ''),
                price_range=product.get('price_range'),
                features=product.get('features', []),
                safety_rating=product.get('safety_rating'),
                recommended_age=product.get('recommended_age')
            ))
        
        # Create recommendation result
        confidence = ConfidenceScore(
            score=min(intent_result.confidence.score, 0.8),  # Cap confidence for recommendations
            reasoning="Based on keyword matching against product database"
        )
        
        result = RecommendationResult(
            products=product_infos,
            reasoning=f"Found {len(product_infos)} products matching your query for baby/child items.",
            confidence=confidence,
            safety_note="Always verify product safety ratings and age appropriateness before purchase."
        )
        
        return result.dict(), "product_recommendation"
        
    except Exception as e:
        raise Exception(f"Product recommendation failed: {str(e)}")


async def handle_informational_query(user_query: UserQuery, intent_result) -> tuple[Dict[str, Any], str]:
    """Handle informational queries about parenting."""
    try:
        # Simple keyword-based knowledge search
        query_lower = user_query.query.lower()
        all_knowledge = knowledge_data.get("knowledge", [])
        
        # Filter knowledge based on keywords
        matching_knowledge = []
        for knowledge in all_knowledge:
            knowledge_text = f"{knowledge.get('title', '')} {knowledge.get('content', '')} {knowledge.get('category', '')}".lower()
            if any(keyword in knowledge_text for keyword in ['parenting', 'baby', 'infant', 'toddler', 'child', 'development']):
                matching_knowledge.append(knowledge)
        
        # Convert to KnowledgeInfo objects
        knowledge_infos = []
        for knowledge in matching_knowledge[:3]:  # Limit to top 3
            knowledge_infos.append(KnowledgeInfo(
                id=knowledge.get('id', ''),
                title=knowledge.get('title', ''),
                category=knowledge.get('category', ''),
                content=knowledge.get('content', ''),
                source=knowledge.get('source'),
                age_relevance=knowledge.get('age_relevance')
            ))
        
        # Create informational result
        confidence = ConfidenceScore(
            score=min(intent_result.confidence.score, 0.7),  # Cap confidence for informational content
            reasoning="Based on keyword matching against parenting knowledge database"
        )
        
        result = InformationalResult(
            knowledge=knowledge_infos,
            summary=f"Found {len(knowledge_infos)} relevant information entries about parenting and child development.",
            confidence=confidence,
            safety_note="This information is for general guidance only and not a substitute for professional advice."
        )
        
        return result.dict(), "informational_response"
        
    except Exception as e:
        raise Exception(f"Informational query failed: {str(e)}")


async def handle_comparison_query(user_query: UserQuery, intent_result) -> tuple[Dict[str, Any], str]:
    """Handle product comparison queries."""
    try:
        # For now, return a limited comparison response
        # In a full implementation, this would extract specific products to compare
        query_lower = user_query.query.lower()
        all_products = products_data.get("products", [])
        
        # Find products mentioned in comparison
        comparison_products = []
        for product in all_products[:3]:  # Limit for demo
            comparison_products.append(ProductInfo(
                id=product.get('id', ''),
                name=product.get('name', ''),
                category=product.get('category', ''),
                price_range=product.get('price_range'),
                features=product.get('features', []),
                safety_rating=product.get('safety_rating'),
                recommended_age=product.get('recommended_age')
            ))
        
        # Create comparison matrix
        comparison_matrix = {
            "categories": list(set(p.category for p in comparison_products)),
            "price_ranges": list(set(p.price_range for p in comparison_products if p.price_range)),
            "safety_ratings": [p.safety_rating for p in comparison_products if p.safety_rating]
        }
        
        confidence = ConfidenceScore(
            score=min(intent_result.confidence.score, 0.6),  # Lower confidence for comparisons
            reasoning="Limited comparison based on available product data"
        )
        
        result = ComparisonResult(
            products=comparison_products,
            comparison_matrix=comparison_matrix,
            recommendation="Consider safety ratings, age appropriateness, and your specific needs when making a decision.",
            confidence=confidence,
            safety_note="Always research products thoroughly and consult professionals when needed."
        )
        
        return result.dict(), "comparison_response"
        
    except Exception as e:
        raise Exception(f"Comparison query failed: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ApiResponse(
            success=False,
            error="An unexpected error occurred. Please try again later.",
            metadata={"error_type": "server_error"}
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)