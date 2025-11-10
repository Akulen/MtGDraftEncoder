from models.linear_regression import LinearRegression

def get_model(model_name: str):
    match model_name:
        case 'LinearRegression':
            return LinearRegression
        case _:
            raise ValueError(f"Unknown model: {model_name}")

def get_model_params(model_name: str):
    import inspect
    return inspect.signature(get_model(model_name).__init__).parameters
