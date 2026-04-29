"""
Safety module for Parent Decision Copilot.

This module handles safety assessments, refusal logic, and professional
consultation recommendations for health-sensitive queries.
"""

import re
from typing import List, Dict, Optional, Tuple
from .schemas import SafetyLevel, SafetyAssessment, HealthSensitiveResponse, IntentType


class SafetyManager:
    """
    Safety management system for handling potentially dangerous queries.
    
    This class provides deterministic safety assessments and appropriate
    responses for health-sensitive or dangerous queries.
    """
    
    def __init__(self):
        """Initialize the safety manager with emergency patterns and responses."""
        self._initialize_emergency_patterns()
        self._initialize_response_templates()
    
    def _initialize_emergency_patterns(self):
        """Initialize patterns for emergency and urgent situations."""
        self.emergency_patterns = {
            'immediate_emergency': [
                r'\b(not breathing|unconscious|unresponsive|seizure|convulsion)\b',
                r'\b(bleeding heavily|severe bleeding|major bleeding)\b',
                r'\b(poison|poisoning|overdose|drug overdose)\b',
                r'\b(choking|can\'t breathe|difficulty breathing|struggling to breathe)\b',
                r'\b(high fever|very high fever|fever over)\b.*\b(baby|infant|toddler|child)\b',
                r'\b(emergency|urgent|immediate|right away|call 911|dial 911)\b'
            ],
            
            'urgent_medical': [
                r'\b(persistent vomiting|continuous vomiting|won\'t stop vomiting)\b',
                r'\b(severe pain|extreme pain|terrible pain)\b',
                r'\b(dehydration|dehydrated|no wet diapers|not urinating)\b',
                r'\b(rash|allergic reaction|swelling|hives)\b.*\b(baby|infant|toddler|child)\b',
                r'\b(fever|temperature)\b.*\b(baby|infant|toddler|child)\b.*\b(days|more than|over)\b',
                r'\b(won\'t eat|refusing food|not eating|not drinking)\b.*\b(baby|infant|toddler|child)\b'
            ],
            
            'medical_concern': [
                r'\b(concern|worried|abnormal|unusual|strange)\b.*\b(baby|infant|toddler|child)\b',
                r'\b(weight|height|growth|developmental|development)\b.*\b(concern|problem|issue|delay)\b',
                r'\b(behavior|behavioral|sleep|eating|feeding)\b.*\b(problem|issue|concern|change)\b',
                r'\b(should I|can I|is it safe|is it normal|is this normal)\b.*\b(baby|infant|toddler|child)\b'
            ]
        }
        
        # Medical professional types by concern category
        self.professional_recommendations = {
            'immediate_emergency': 'emergency medical services (call 911 or go to nearest emergency room)',
            'urgent_medical': 'pediatrician or urgent care center',
            'medical_concern': 'pediatrician or family doctor',
            'general_health': 'pediatrician or family doctor',
            'developmental': 'pediatrician or developmental specialist'
        }
    
    def _initialize_response_templates(self):
        """Initialize response templates for different safety levels."""
        self.response_templates = {
            SafetyLevel.DANGEROUS: {
                'refusal': "I cannot provide medical advice for this situation. This appears to be a medical emergency that requires immediate professional attention.",
                'guidance': "Please seek emergency medical care immediately by calling emergency services or going to the nearest emergency room.",
                'professional': 'emergency medical services'
            },
            
            SafetyLevel.CAUTION: {
                'refusal': "I cannot provide medical advice for health concerns. Your question involves medical matters that require professional evaluation.",
                'guidance': "Please consult with a qualified healthcare professional who can properly assess your child's specific situation.",
                'professional': 'pediatrician or qualified healthcare provider'
            },
            
            SafetyLevel.SAFE: {
                'refusal': "I cannot provide medical advice. For any health-related concerns, please consult with a qualified healthcare professional.",
                'guidance': "While I can provide general parenting information, medical questions should always be directed to healthcare professionals.",
                'professional': 'pediatrician or family doctor'
            }
        }
    
    def _match_emergency_patterns(self, query: str) -> Tuple[str, float]:
        """
        Match query against emergency patterns to determine urgency level.
        
        Args:
            query: User query text
            
        Returns:
            Tuple of (urgency_category, match_strength)
        """
        query_lower = query.lower()
        
        for category, patterns in self.emergency_patterns.items():
            match_count = 0
            total_patterns = len(patterns)
            
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    match_count += 1
            
            if match_count > 0:
                match_strength = match_count / total_patterns
                return category, match_strength
        
        return 'general_health', 0.0
    
    def _determine_safety_level(self, query: str, intent_type: IntentType) -> SafetyLevel:
        """
        Determine the safety level of a query.
        
        Args:
            query: User query text
            intent_type: Classified intent type
            
        Returns:
            Safety level classification
        """
        # If already classified as health-sensitive, perform detailed safety assessment
        if intent_type == IntentType.HEALTH_SENSITIVE_QUERY:
            urgency_category, match_strength = self._match_emergency_patterns(query)
            
            if urgency_category == 'immediate_emergency' or match_strength >= 0.5:
                return SafetyLevel.DANGEROUS
            elif urgency_category == 'urgent_medical' or match_strength >= 0.3:
                return SafetyLevel.CAUTION
            else:
                return SafetyLevel.SAFE
        
        # For other intents, check for any medical keywords
        medical_keywords = [
            'doctor', 'medical', 'medicine', 'medication', 'prescription',
            'treatment', 'therapy', 'symptom', 'diagnosis', 'condition'
        ]
        
        query_lower = query.lower()
        medical_keyword_count = sum(1 for keyword in medical_keywords if keyword in query_lower)
        
        if medical_keyword_count >= 2:
            return SafetyLevel.CAUTION
        elif medical_keyword_count >= 1:
            return SafetyLevel.SAFE
        
        return SafetyLevel.SAFE
    
    def _get_professional_recommendation(self, urgency_category: str) -> str:
        """
        Get appropriate professional recommendation based on urgency.
        
        Args:
            urgency_category: Category of urgency from pattern matching
            
        Returns:
            Recommended professional type
        """
        return self.professional_recommendations.get(
            urgency_category,
            self.professional_recommendations['general_health']
        )
    
    def assess_safety(self, query: str, intent_type: IntentType) -> SafetyAssessment:
        """
        Perform comprehensive safety assessment of user query.
        
        Args:
            query: User query text
            intent_type: Classified intent type
            
        Returns:
            Safety assessment with recommendations
        """
        safety_level = self._determine_safety_level(query, intent_type)
        
        # Determine if query is safe to answer
        is_safe = safety_level == SafetyLevel.SAFE and intent_type != IntentType.HEALTH_SENSITIVE_QUERY
        
        # Generate refusal reason and recommended action
        refusal_reason = None
        recommended_action = None
        
        if not is_safe:
            urgency_category, _ = self._match_emergency_patterns(query)
            professional = self._get_professional_recommendation(urgency_category)
            
            if safety_level == SafetyLevel.DANGEROUS:
                refusal_reason = "This query involves a potential medical emergency that requires immediate professional attention."
                recommended_action = f"Seek immediate medical help from {professional}."
            elif safety_level == SafetyLevel.CAUTION:
                refusal_reason = "This query involves health-sensitive matters that require professional medical evaluation."
                recommended_action = f"Consult with {professional} for proper assessment."
            else:
                refusal_reason = "This query involves medical matters that should be addressed by healthcare professionals."
                recommended_action = f"Contact {professional} for guidance."
        
        return SafetyAssessment(
            is_safe=is_safe,
            safety_level=safety_level,
            refusal_reason=refusal_reason,
            recommended_action=recommended_action
        )
    
    def create_health_sensitive_response(self, query: str, safety_level: SafetyLevel) -> HealthSensitiveResponse:
        """
        Create appropriate response for health-sensitive queries.
        
        Args:
            query: User query text
            safety_level: Assessed safety level
            
        Returns:
            Health-sensitive response with professional recommendations
        """
        urgency_category, _ = self._match_emergency_patterns(query)
        professional = self._get_professional_recommendation(urgency_category)
        
        template = self.response_templates[safety_level]
        
        # Determine urgency level for response
        urgency_level = None
        if safety_level == SafetyLevel.DANGEROUS:
            urgency_level = "Emergency - Immediate attention required"
        elif safety_level == SafetyLevel.CAUTION:
            urgency_level = "Urgent - Professional consultation recommended"
        
        # General non-medical guidance that's safe to provide
        general_guidance = None
        if safety_level == SafetyLevel.SAFE:
            general_guidance = "For general parenting questions not related to medical concerns, I'm happy to help with product recommendations, developmental information, and parenting advice based on established guidelines."
        
        return HealthSensitiveResponse(
            refusal_message=template['refusal'],
            recommended_professional=professional,
            urgency_level=urgency_level,
            general_guidance=general_guidance
        )
    
    def is_safe_to_proceed(self, safety_assessment: SafetyAssessment) -> bool:
        """
        Determine if it's safe to proceed with answering the query.
        
        Args:
            safety_assessment: Safety assessment result
            
        Returns:
            True if safe to proceed, False otherwise
        """
        return safety_assessment.is_safe and safety_assessment.safety_level == SafetyLevel.SAFE
    
    def get_safety_disclaimer(self) -> str:
        """
        Get standard safety disclaimer for medical-related queries.
        
        Returns:
            Safety disclaimer text
        """
        return (
            "Important: I am an AI assistant and not a medical professional. "
            "For any health concerns about your child, always consult with qualified "
            "healthcare providers such as pediatricians, family doctors, or emergency services. "
            "In case of emergency, call emergency services immediately."
        )


def create_safety_manager() -> SafetyManager:
    """
    Factory function to create a SafetyManager instance.
    
    Returns:
        Configured SafetyManager instance
    """
    return SafetyManager()