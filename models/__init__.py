from models._types import DraftWRPredictor

from models.draft_attention import DraftTransformer2
from models.draft_graph import DraftGraph2
from models.baselines import LinearRegression, Constant, Heuristic

def get_model(model_name: str) -> type[DraftWRPredictor]:
    match model_name:
        case 'DraftTransformer2':
            return DraftTransformer2
        case 'DraftGraph2':
            return DraftGraph2
        case 'LinearRegression':
            return LinearRegression
        case 'Constant':
            return Constant
        case 'Heuristic':
            return Heuristic
        case _:
            raise ValueError(f"Unknown model: {model_name}")

def get_model_params(model_name: str):
    import inspect
    return inspect.signature(get_model(model_name).__init__).parameters
