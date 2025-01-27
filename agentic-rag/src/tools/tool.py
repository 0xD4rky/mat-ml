import os
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, ConfigDict
from markitdown import MarkItDown
from chonkie import SemanticChunker
from sentence_transformer import sentence_transformer

class DocumentSearchToolInput(BaseModel):
    query: str = Field(..., description="Query to search the document.")

class DocumentSearchTool(BaseTool):

    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput

    model_config = ConfigDict(extra="allow")

