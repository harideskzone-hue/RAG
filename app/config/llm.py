import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically load environment variables from .env.local or .env
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if (_PROJECT_ROOT / ".env.local").exists():
    load_dotenv(_PROJECT_ROOT / ".env.local", override=False)
if (_PROJECT_ROOT / ".env").exists():
    load_dotenv(_PROJECT_ROOT / ".env", override=False)

def get_model_for_role(role: str) -> tuple[str, str]:
    """
    Returns (provider, model_id) for a specific VISTA role.
    Defaults to LLM_PROVIDER and LLM_MODEL if role-specific config is not set.
    """
    provider = os.getenv("LLM_PROVIDER", "groq")
    default_model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    
    role_mapping = {
        "intent": os.getenv("VISTA_INTENT_MODEL", default_model),
        "planner": os.getenv("VISTA_PLANNER_MODEL", default_model),
        "reasoning": os.getenv("VISTA_REASONING_MODEL", default_model),
        "response": os.getenv("VISTA_RESPONSE_MODEL", default_model),
        "vision": os.getenv("VISTA_VISION_MODEL", "qwen/qwen3.6-27b"),
    }
    
    model = role_mapping.get(role, default_model)
    return provider, model
