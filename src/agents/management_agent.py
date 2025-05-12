import os
from openai import OpenAI
import json

class ManagementAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_management_advice(self, diagnosis: str, case_data: dict) -> str:
        prompt = (
            f"You are a clinical reasoning assistant. Provide concise, evidence-based management and treatment advice for the following diagnosis and case context. "
            f"If the diagnosis is highly likely, focus on management. If there is uncertainty, mention what further steps are needed.\n"
            f"Diagnosis: {diagnosis}\nCase: {json.dumps(case_data)}"
        )
        messages = [
            {"role": "system", "content": "You are a clinical reasoning assistant."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[ManagementAgent error: {str(e)}]" 