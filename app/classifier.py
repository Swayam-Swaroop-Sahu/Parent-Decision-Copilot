"""
Intent classification module for Parent Decision Copilot.

This module handles the classification of user queries into intent types
using rule-based and pattern-matching approaches for deterministic behavior.
"""

import re
from typing import List, Dict, Tuple
from .schemas import IntentType, IntentClassification, ConfidenceScore, SafetyLevel


class IntentClassifier:
    """
    Deterministic intent classifier using rule-based patterns.
    
    This approach ensures consistent, predictable behavior without
    relying on external ML models that might have safety concerns.
    """
    
    def __init__(self):
        """Initialize the classifier with intent patterns."""
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize regex patterns for each intent type."""
        self.intent_patterns = {
            IntentType.PRODUCT_RECOMMENDATION: [
                r'\b(recommend|suggest|what|which|best|good|top)\b.*\b(baby|infant|toddler|child|kid)\b.*\b(product|item|gear|toy|stroller|car seat|diaper|bottle|crib|high chair)\b',
                r'\b(need|want|looking for|searching for)\b.*\b(baby|infant|toddler|child|kid)\b.*\b(product|item|gear)\b',
                r'\b(buy|purchase|get|find)\b.*\b(baby|infant|toddler|child|kid)\b.*\b(product|item|gear)\b',
                r'\b(product|item|gear)\b.*\b(for|suitable|good)\b.*\b(baby|infant|toddler|child|kid)\b'
            ],
            
            IntentType.INFORMATIONAL_QUERY: [
                r'\b(how|what|when|where|why|tell me|explain|describe)\b.*\b(baby|infant|toddler|child|kid|parenting|child development)\b',
                r'\b(information|advice|tips|guidance|help)\b.*\b(parenting|baby|infant|toddler|child|kid)\b',
                r'\b(learn|understand|know about)\b.*\b(baby|infant|toddler|child|kid|parenting)\b',
                r'\b(development|milestone|behavior|growth|care)\b.*\b(baby|infant|toddler|child|kid)\b'
            ],
            
            IntentType.COMPARISON_QUERY: [
                r'\b(compare|comparison|difference|versus|vs|better than|which is better)\b.*\b(baby|infant|toddler|child|kid)\b',
                r'\b(between|among)\b.*\b(product|item|gear|brand|type)\b.*\b(baby|infant|toddler|child|kid)\b',
                r'\b(pros and cons|advantages|disadvantages)\b.*\b(baby|infant|toddler|child|kid)\b.*\b(product|item|gear)\b'
            ],
            
            IntentType.HEALTH_SENSITIVE_QUERY: [
                r'\b(medical|doctor|physician|pediatrician|hospital|emergency|urgent)\b',
                r'\b(fever|temperature|sick|illness|disease|infection|pain|hurt|ache)\b.*\b(baby|infant|toddler|child|kid)\b',
                r'\b(medicine|medication|drug|prescription|treatment|therapy)\b.*\b(baby|infant|toddler|child|kid)\b',
                r'\b(symptom|diagnosis|condition|disorder|problem|issue)\b.*\b(baby|infant|toddler|child|kid)\b',
                r'\b(allergic|allergy|reaction|rash|breathing|breath|respiratory)\b.*\b(baby|infant|toddler|child|kid)\b',
                r'\b(weight|height|growth|developmental delay|concern)\b.*\b(baby|infant|toddler|child|kid)\b',
                r'\b(should I|can I|is it safe|is it normal)\b.*\b(baby|infant|toddler|child|kid)\b.*\b(see doctor|go to hospital|medical)\b'
            ]
        }
        
        # Safety keywords that indicate higher risk
        self.safety_keywords = {
            SafetyLevel.DANGEROUS: [
                'emergency', 'urgent', 'severe', 'critical', 'life threatening',
                'unconscious', 'not breathing', 'bleeding', 'poison', 'overdose',
                'high fever', 'difficulty breathing', 'seizure', 'convulsion'
            ],
            SafetyLevel.CAUTION: [
                'concern', 'worried', 'abnormal', 'unusual', 'persistent',
                'worsening', 'not getting better', 'chronic', 'recurring'
            ]
        }
    
    def _calculate_pattern_score(self, text: str, patterns: List[str]) -> float:
        """
        Calculate how well text matches a list of patterns.
        
        Args:
            text: Input text to analyze
            patterns: List of regex patterns to match against
            
        Returns:
            Score between 0 and 1 indicating match strength
        """
        if not patterns:
            return 0.0
        
        match_count = 0
        total_patterns = len(patterns)
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                match_count += 1
        
        return match_count / total_patterns
    
    def _assess_safety_level(self, text: str) -> SafetyLevel:
        """
        Assess the safety level of the query.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Safety level classification
        """
        text_lower = text.lower()
        
        # Check for dangerous keywords first
        for keyword in self.safety_keywords[SafetyLevel.DANGEROUS]:
            if keyword in text_lower:
                return SafetyLevel.DANGEROUS
        
        # Check for caution keywords
        for keyword in self.safety_keywords[SafetyLevel.CAUTION]:
            if keyword in text_lower:
                return SafetyLevel.CAUTION
        
        return SafetyLevel.SAFE
    
    def _calculate_confidence(self, scores: Dict[IntentType, float]) -> ConfidenceScore:
        """
        Calculate confidence score for intent classification.
        
        Args:
            scores: Dictionary of intent types and their match scores
            
        Returns:
            Confidence score object with reasoning
        """
        if not scores:
            return ConfidenceScore(score=0.0, reasoning="No scores available")
        
        max_score = max(scores.values())
        max_intent = max(scores, key=scores.get)
        
        # Calculate confidence based on how much the top score stands out
        sorted_scores = sorted(scores.values(), reverse=True)
        
        if len(sorted_scores) == 1:
            confidence = max_score
            reasoning = f"Single intent match: {max_intent.value}"
        else:
            second_best = sorted_scores[1]
            confidence_gap = max_score - second_best
            confidence = min(max_score, 0.5 + confidence_gap)
            reasoning = f"Top intent: {max_intent.value} (gap: {confidence_gap:.2f})"
        
        # Additional confidence boost for clear matches
        if max_score > 0.8:
            confidence = min(confidence + 0.1, 1.0)
            reasoning += " - strong pattern match"
        elif max_score < 0.3:
            confidence = max(confidence - 0.1, 0.0)
            reasoning += " - weak pattern match"
        
        return ConfidenceScore(score=confidence, reasoning=reasoning)
    
    def classify_intent(self, query: str) -> IntentClassification:
        """
        Classify user query into intent type.
        
        Args:
            query: User query text
            
        Returns:
            Intent classification result with confidence and safety assessment
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Normalize query text
        normalized_query = query.strip().lower()
        
        # Calculate pattern scores for each intent
        intent_scores = {}
        for intent_type, patterns in self.intent_patterns.items():
            score = self._calculate_pattern_score(normalized_query, patterns)
            intent_scores[intent_type] = score
        
        # Determine the best matching intent
        best_intent = max(intent_scores, key=intent_scores.get)
        best_score = intent_scores[best_intent]
        
        # If no strong match found, default to informational
        if best_score < 0.2:
            best_intent = IntentType.INFORMATIONAL_QUERY
            best_score = 0.1
        
        # Calculate confidence
        confidence = self._calculate_confidence(intent_scores)
        
        # Assess safety level
        safety_level = self._assess_safety_level(query)
        
        # If safety level is dangerous or caution, override to health-sensitive
        if safety_level in [SafetyLevel.DANGEROUS, SafetyLevel.CAUTION]:
            best_intent = IntentType.HEALTH_SENSITIVE_QUERY
            confidence.reasoning += " - safety override applied"
        
        return IntentClassification(
            intent=best_intent,
            confidence=confidence,
            safety_level=safety_level
        )
    
    def get_intent_explanation(self, intent: IntentType) -> str:
        """
        Get human-readable explanation for an intent type.
        
        Args:
            intent: Intent type to explain
            
        Returns:
            Explanation string
        """
        explanations = {
            IntentType.PRODUCT_RECOMMENDATION: "User is looking for product recommendations or suggestions for baby/child items",
            IntentType.INFORMATIONAL_QUERY: "User is seeking general information or advice about parenting and child development",
            IntentType.COMPARISON_QUERY: "User wants to compare different products, brands, or options",
            IntentType.HEALTH_SENSITIVE_QUERY: "User is asking about health-related matters that require professional medical advice"
        }
        
        return explanations.get(intent, "Unknown intent type")


def create_classifier() -> IntentClassifier:
    """
    Factory function to create an IntentClassifier instance.
    
    Returns:
        Configured IntentClassifier instance
    """
    return IntentClassifier()