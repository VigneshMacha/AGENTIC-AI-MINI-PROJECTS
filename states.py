import os
from langgraph.graph import MessagesState 
from pydantic import BaseModel,field_validator
from typing import TypedDict

class staate(TypedDict):
    topic:str
    summary:str
    score:int


class state(BaseModel):
    topic:str
    summary:str=""
    score:int
    @field_validator
    def check_score(cls,v):
        if v<0:
            raise ValueError("Score must be positive")


from dataclasses import dataclass,field

@dataclass
class state:
    topic:str=""
    summary:str=""
    messages:list=field(default_factory=list)

class State(MessagesState):
    user_name:str
    language:str=""
    topic:str=""
    
