from langflow import CustomComponent
from langflow.field_typing import Data, Text
from typing import Optional
import requests
import json
import logging

logger = logging.getLogger(__name__)


class MentalHealthRAGComponent(CustomComponent):
    """Custom Langflow component for Mental Health RAG system"""

    display_name = "Mental Health RAG"
    description = "RAG system for mental health support based on Reddit data"

    def build_config(self):
        return {
            "api_url": {
                "display_name": "API URL",
                "field_type": "str",
                "required": True,
                "value": "http://localhost:8000/query"
            },
            "user_query": {
                "display_name": "User Query",
                "field_type": "str",
                "required": True,
                "multiline": True
            },
            "user_id": {
                "display_name": "User ID",
                "field_type": "str",
                "required": False,
                "value": "anonymous"
            }
        }

    def build(
            self,
            api_url: str,
            user_query: str,
            user_id: Optional[str] = None
    ) -> Data:
        """Process query through Mental Health RAG system"""

        try:
            # Prepare request payload
            payload = {
                "query": user_query,
                "user_id": user_id or "anonymous"
            }

            # Make API call
            response = requests.post(
                api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            # Format response
            formatted_response = self._format_response(result)

            return Data(data={
                "response": formatted_response,
                "raw_result": result,
                "is_crisis": result.get("is_crisis", False),
                "sources_used": result.get("sources_used", 0)
            })

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            error_response = "I'm sorry, I'm having trouble connecting right now. If you're in crisis, please call 988 (Suicide & Crisis Lifeline) immediately."

            return Data(data={
                "response": error_response,
                "raw_result": {},
                "is_crisis": True,  # Assume crisis for safety
                "sources_used": 0
            })

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            error_response = "I apologize for the technical difficulty. Please reach out to a mental health professional if you need support."

            return Data(data={
                "response": error_response,
                "raw_result": {},
                "is_crisis": False,
                "sources_used": 0
            })

    def _format_response(self, result: dict) -> str:
        """Format the response with crisis warnings if needed"""
        response = result.get("response", "No response available")
        is_crisis = result.get("is_crisis", False)
        sources_used = result.get("sources_used", 0)

        # Add crisis warning if detected
        if is_crisis:
            warning = """🚨 CRISIS SUPPORT RESOURCES 🚨
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• Emergency Services: 911

"""
            response = warning + response

        # Add source information
        footer = f"\n\n💬 This response is based on {sources_used} community discussions from Reddit mental health communities. This is for informational purposes only and not a substitute for professional mental health care."

        return response + footer


class CrisisDetectionComponent(CustomComponent):
    """Component specifically for crisis detection"""

    display_name = "Crisis Detection"
    description = "Detect crisis indicators in user messages"

    def build_config(self):
        return {
            "user_message": {
                "display_name": "User Message",
                "field_type": "str",
                "required": True,
                "multiline": True
            },
            "threshold": {
                "display_name": "Crisis Threshold",
                "field_type": "float",
                "required": True,
                "value": 0.7
            }
        }

    def build(self, user_message: str, threshold: float = 0.7) -> Data:
        """Detect crisis indicators in user message"""

        crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'self harm', 'cutting',
            'overdose', 'no point', 'better off dead', 'want to die', 'ending it'
        ]

        message_lower = user_message.lower()
        crisis_score = 0.0

        # Keyword detection
        for keyword in crisis_keywords:
            if keyword in message_lower:
                crisis_score += 0.2

        # Pattern detection
        import re
        crisis_patterns = [
            r'\bi\s+want\s+to\s+die\b',
            r'\bkill\s+myself\b',
            r'\bend\s+it\s+all\b',
            r'\bno\s+point\s+in\s+living\b'
        ]

        for pattern in crisis_patterns:
            if re.search(pattern, message_lower):
                crisis_score += 0.3

        is_crisis = crisis_score >= threshold

        return Data(data={
            "is_crisis": is_crisis,
            "crisis_score": crisis_score,
            "alert_level": "HIGH" if is_crisis else "LOW",
            "message": user_message
        })


# Langflow flow configuration
LANGFLOW_CONFIG = {
    "name": "Mental Health Support Chatbot",
    "description": "AI chatbot for mental health support using Reddit community data",
    "nodes": [
        {
            "id": "input_node",
            "type": "TextInput",
            "data": {
                "name": "User Input",
                "value": ""
            },
            "position": {"x": 100, "y": 100}
        },
        {
            "id": "crisis_detection",
            "type": "CrisisDetectionComponent",
            "data": {
                "user_message": "{input_node.text}",
                "threshold": 0.7
            },
            "position": {"x": 300, "y": 100}
        },
        {
            "id": "rag_component",
            "type": "MentalHealthRAGComponent",
            "data": {
                "api_url": "http://localhost:8000/query",
                "user_query": "{input_node.text}",
                "user_id": "langflow_user"
            },
            "position": {"x": 500, "y": 100}
        },
        {
            "id": "output_node",
            "type": "TextOutput",
            "data": {
                "text": "{rag_component.response}"
            },
            "position": {"x": 700, "y": 100}
        }
    ],
    "edges": [
        {
            "source": "input_node",
            "target": "crisis_detection"
        },
        {
            "source": "input_node",
            "target": "rag_component"
        },
        {
            "source": "rag_component",
            "target": "output_node"
        }
    ]
}