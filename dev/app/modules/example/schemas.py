from pydantic import BaseModel


class ExampleOut(BaseModel):
    id: int
    name: str
