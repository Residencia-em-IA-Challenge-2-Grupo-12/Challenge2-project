# Camada de apresentação: rotas HTTP.
from fastapi import APIRouter

from . import service
from .schemas import ExampleOut

router = APIRouter(prefix="/examples", tags=["example"])


@router.get("", response_model=list[ExampleOut])
def list_examples():
    return service.list_examples()
