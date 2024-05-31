from typing import List
from pydantic import BaseModel

class Query(BaseModel):
    text: str


class Document(BaseModel):
    text: List[str]

