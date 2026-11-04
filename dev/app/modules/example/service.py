# Camada de negócio: regras da aplicação.
from . import repository
from .schemas import ExampleOut


def list_examples() -> list[ExampleOut]:
    return [ExampleOut(**row) for row in repository.list_all()]
