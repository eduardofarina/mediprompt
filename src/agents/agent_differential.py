import json
from typing import Dict, Any, List
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DifferentialDiagnosisAgent:
    """
    This agent generates up to 3 plausible differential diagnoses and suggests next steps (questions or tests)
    based on the available case data. If data is insufficient, it asks for the most discriminating missing data.
    """
    def __init__(self):
        pass

    def get_differential_and_next_steps(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        # Prepare a summary of the case
        case_summary = "\n".join([f"{k}: {v}" for k, v in case_data.items() if v])
        missing_fields = [k for k, v in case_data.items() if not v]
        
        prompt = f"""
        You are a clinical reasoning assistant. Given the following patient data, generate up to 3 plausible differential diagnoses (with brief reasoning for each), and suggest the most useful next question or diagnostic test to narrow the differential. If data is insufficient, ask for the most discriminating missing information or suggest a test. Never return an empty differential.
        
        Patient data:
        {case_summary if case_summary else '[No data provided]'}
        
        Missing fields: {', '.join(missing_fields) if missing_fields else 'None'}
        
        Respond in this JSON format:
        {{
            "differential": [
                {{"diagnosis": "...", "reasoning": "..."}},
                ...
            ],
            "next_steps": ["...", "..."],
            "ask_user": "..."  // A concise question for the user or a test to order
        }}
        """
        messages = [
            {"role": "system", "content": "You are a clinical reasoning assistant."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
            )
            content = response.choices[0].message.content
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
                return result
        except Exception as e:
            return {
                "differential": [
                    {"diagnosis": "Unable to generate differential", "reasoning": str(e)}
                ],
                "next_steps": ["Please provide more information."],
                "ask_user": "What is the most important missing information?"
            } 