from fastapi import APIRouter

from SearchAPI.app.api.get_document import get_document_response
from SearchAPI.app.api.models import Document, Query



searcher  = APIRouter()




@searcher.post('/get_document',response_model=Document)
async def get_document(payload: Query):
    return get_document_response(payload.text)
