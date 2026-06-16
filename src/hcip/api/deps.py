import json
from pathlib import Path
from typing import Any

from hcip.modeling import load_model

MODELS_DIR = Path("data/models")

class ModelRegistry:
    def __init__(self):
        self.breach_model = None
        self.demand_model = None
        self.wait_time_model = None
        self.metadata = {}
    
    def load_all(self):
        if (MODELS_DIR / "breach_model.pkl").exists():
            self.breach_model = load_model(MODELS_DIR / "breach_model.pkl")
        if (MODELS_DIR / "demand_model.pkl").exists():
            self.demand_model = load_model(MODELS_DIR / "demand_model.pkl")
        if (MODELS_DIR / "wait_time_model.pkl").exists():
            self.wait_time_model = load_model(MODELS_DIR / "wait_time_model.pkl")
        if (MODELS_DIR / "metadata.json").exists():
            with open(MODELS_DIR / "metadata.json", "r") as f:
                self.metadata = json.load(f)

# Global registry instance
registry = ModelRegistry()

def get_model_registry() -> ModelRegistry:
    return registry
