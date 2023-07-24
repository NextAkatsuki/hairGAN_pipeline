from pydantic import BaseModel

class PredictResult(BaseModel):
    name: str
    prob: float