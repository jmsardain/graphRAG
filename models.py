from openai import OpenAI
import os

class OpenAIClient:
    def __init__(self, model="gpt-4o-mini", temperature=0.0):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a research assistant helping with scientific literature analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )
        return resp.choices[0].message.content.strip()
