import os
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, ConfigDict
from markitdown import MarkItDown
from chonkie import SemanticChunker
from sentence_transformers import SentenceTransformer

class DocumentSearchToolInput(BaseModel):
    query: str = Field(..., description="Query to search the document.")

class DocumentSearchTool(BaseTool):

    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput

    model_config = ConfigDict(extra="allow")

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self.encoder = SentenceTransformer("minishlab/potion-base-8M")
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        self._process_document()

    def _extract_text(self) -> str:
        md = MarkItDown()
        result = md.convert(self.file_path)
        return result.text_content

    def _create_chunks(self, raw_text: str) -> list:
        chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=512,
            min_sentences=1
        )
        return chunker.chunk(raw_text)
    
    def _process_document(self):
        raw_text = self._extract_text()
        self.chunks = self._create_chunks(raw_text)
        
        vectors = []
        for chunk in self.chunks:
            vector = self.encoder.encode(chunk.text)
            vectors.append(vector)
        
        vectors = np.array(vectors).astype('float32')
        self.index.add(vectors)

    def _run(self, query: str) -> str:
        query_vector = self.encoder.encode(query).reshape(1, -1).astype('float32')
        k = 3
        distances, indices = self.index.search(query_vector, k)
        
        relevant_chunks = [self.chunks[idx].text for idx in indices[0]]
        return "\n___\n".join(relevant_chunks)

def test_document_searcher():
    pdf_path = "/Users/darky/Documents/mat-ml/deepseek_R1.pdf"
    searcher = DocumentSearchTool(file_path=pdf_path)
    result = searcher._run("What is GRPO?")
    print("Search Results:", result)

if __name__ == "__main__":
    test_document_searcher()
