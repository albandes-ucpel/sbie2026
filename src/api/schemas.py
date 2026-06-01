from pydantic import BaseModel

class Query(BaseModel):
    query: str
    k: int = 5
    class_id: str | None = None
