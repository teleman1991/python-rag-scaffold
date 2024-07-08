from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type
from typing import List, Dict, Any
from fastapi import APIRouter, Form
from openai import OpenAI, BadRequestError
from dataclasses import dataclass
import turbopuffer as tpuf
import cohere
import os

router = APIRouter()
client = OpenAI()
tpuf.api_key = os.getenv("TURBOPUFFER_API_KEY")
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

@dataclass
class Retrieval:
    text: str                   # The text of the chunk
    embedding: List[float]       # The embedding of the chunk
    vectorID: str               # The unique vectorID of the chunk, to identify it in vectordb
    fileID: str                  # A unique fileID of the chunk, for relational db purposes
    distance: float              # Cosine similarity score
    reranker_score: float        # Reranker relevance score


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6), retry=retry_if_not_exception_type(BadRequestError))
def get_embeddings(text_or_tokens, model="text-embedding-3-small"):
    return client.embeddings.create(input=text_or_tokens, model=model, dimensions=512).data[0].embedding

@router.post("/retrieve")
async def get_context(queries: List[str] = Form(...), namespace: str = Form(...)) -> List[List[Dict[str, Any]]]:
    response: List[List[Dict[str, Any]]] = []
    ns = tpuf.Namespace(namespace)
    
    for query in queries:
        # Step 1: Convert query to embedding
        embedding = get_embeddings(query)

        # Step 2: Get top 10 relevant chunks via vectordb + store cosine similarity scores
        vectors = ns.query(
            vector=embedding,
            distance_metric='cosine_distance',
            top_k=10,
            include_attributes=['text', 'fileID']
        )
        
        # Prepare documents for reranking
        docs = [vector.attributes["text"] for vector in vectors]
        
        if not docs:
            response.append([])
            continue

        # Step 3: Rerank chunks + store reranker score
        rerank_results = co.rerank(model="rerank-english-v3.0", query=query, documents=docs, top_n=10, return_documents=True)
        
        retrieved_chunks: List[Retrieval] = []
        for idx, result in enumerate(rerank_results.results):
            retrieved_chunks.append(Retrieval(
                text=result.document.text,
                embedding=vectors[result.index].vector,
                vectorID=vectors[result.index].id,
                fileID=vectors[result.index].attributes['fileID'],
                distance=vectors[result.index].dist,
                reranker_score=result.relevance_score
            ))
        
        # Sort the chunks by reranker score
        retrieved_chunks.sort(key=lambda x: x.reranker_score, reverse=True)

        # Convert to dictionary for the response
        response.append([{
            "text": chunk.text,
            "vectorID": chunk.vectorID,
            "fileID": chunk.fileID,
            "distance": chunk.distance,
            "reranker_score": chunk.reranker_score
        } for chunk in retrieved_chunks])

    return response