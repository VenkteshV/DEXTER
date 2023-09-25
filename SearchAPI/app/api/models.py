from typing import List,Any
from pydantic import BaseModel

class Query(BaseModel):
    text: str


class Document(BaseModel):
    text: List[str]

