from models._types import DraftWRPredictor

from models.draft_attention import DraftAttention, DraftTransformer
from models.draft_attention import DraftGraph
# from models.draft_graph import DraftGraph
from models.linear_regression import LinearRegression

def get_model(model_name: str):
    match model_name:
        case 'DraftAttention':
            return DraftAttention
        case 'DraftTransformer':
            return DraftTransformer
        case 'DraftGraph':
            return DraftGraph
        case 'LinearRegression':
            return LinearRegression
        case _:
            raise ValueError(f"Unknown model: {model_name}")

def get_model_params(model_name: str):
    import inspect
    return inspect.signature(get_model(model_name).__init__).parameters
