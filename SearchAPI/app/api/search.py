from typing import List,Any
from fastapi import Header, APIRouter
from app.api.models import Query, Document
from app.api.get_document import  get_document_response


searcher  = APIRouter()




@searcher.post('/get_document',response_model=Document)
async def get_document(payload: Query):
    return get_document_response(payload.text)
