from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
import requests
from jinja2 import Template
from llama_client import generate_summary

# Environment variables needed
# OPENAI_API_KEY=your-secret-api-key
# OPENAI_MODEL=gpt-4
# PREPROCESSOR_URL=http://preprocessor:8001/metrics-summary

app = FastAPI()

PREPROCESSOR_URL = os.getenv("PREPROCESSOR_URL", "http://preprocessor:8001/metrics-summary")

with open("prompt_template.j2", "r") as f:
    prompt_template = Template(f.read())

@app.get("/summarize")
def summarize():
    try:
        response = requests.get(PREPROCESSOR_URL)
        if response.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "Failed to fetch from preprocessor"})

        metrics = response.json()
        prompt = prompt_template.render(**metrics)
        summary = generate_summary(prompt)

        return {
            "prompt": prompt,
            "summary": summary
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
