from fastapi import APIRouter
from typing import List, Dict, Any
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY
from llm import Llm, MODEL_PROVIDER, OPENAI_MODELS, ANTHROPIC_MODELS, GEMINI_MODELS

router = APIRouter()


@router.get("/available-models")
def get_available_models() -> Dict[str, Any]:
    """
    Returns available models based on which API keys are configured.
    """
    available_models = []
    providers = []
    
    # Check OpenAI models
    if OPENAI_API_KEY:
        providers.append("openai")
        for model in OPENAI_MODELS:
            available_models.append({
                "id": model.value,
                "name": model.name,
                "provider": "openai",
                "display_name": model.name.replace("_", " ").title()
            })
    
    # Check Anthropic models  
    if ANTHROPIC_API_KEY:
        providers.append("anthropic")
        for model in ANTHROPIC_MODELS:
            available_models.append({
                "id": model.value,
                "name": model.name,
                "provider": "anthropic", 
                "display_name": model.name.replace("_", " ").title()
            })
    
    # Check Gemini models
    if GEMINI_API_KEY:
        providers.append("gemini")
        for model in GEMINI_MODELS:
            available_models.append({
                "id": model.value,
                "name": model.name,
                "provider": "gemini",
                "display_name": model.name.replace("_", " ").title()
            })
    
    return {
        "models": available_models,
        "providers": providers,
        "has_models": len(available_models) > 0
    }


@router.get("/default-models")
def get_default_models() -> Dict[str, Any]:
    """
    Returns the default model selection based on available API keys.
    This mimics the current logic in generate_code.py
    """
    models = []
    
    # Primary model selection logic (from generate_code.py)
    if ANTHROPIC_API_KEY:
        models.append("claude-3-7-sonnet-20250219")  # Primary Claude model
    
    if OPENAI_API_KEY:
        models.append("gpt-4o-2024-11-20")  # Primary OpenAI model
    
    # Fallback if only one provider
    if len(models) == 0:
        if GEMINI_API_KEY:
            models.append("gemini-2.0-flash")
    
    return {
        "models": models,
        "count": len(models)
    }
