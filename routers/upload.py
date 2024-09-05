from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type
from fastapi import APIRouter, UploadFile, File, Form
from chunkipy import TextChunker, TokenEstimator
from openai import OpenAI, BadRequestError
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import List, Dict

import turbopuffer as tpuf
import numpy as np
import tiktoken
import time
import nltk
import fitz  
import os
import re

load_dotenv()
client = OpenAI()
router = APIRouter()
nltk.download('punkt')
tpuf.api_key = os.getenv("TURBOPUFFER_API_KEY")
@dataclass
class Chunk:
    text: str                   # The text of the chunk
    embedding: List[float]       # The embedding of the chunk
    vectorID: str               # The unique vectorID of the chunk, to identify it in vectordb
    fileID: str                  # A unique fileID of the chunk, for relational db purposes

class OpenAITokenEstimator(TokenEstimator):
    def __init__(self):
        self.tiktoken_tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")
    def estimate_tokens(self, text):
        return len(self.tiktoken_tokenizer.encode(text))

openai_token_estimator = OpenAITokenEstimator()
download_np_libraries = TextChunker(512, tokens=True, overlap_percent=10, token_estimator=OpenAITokenEstimator(), split_strategies=[sent_tokenize])

def extract_text_from_pdf(file: UploadFile) -> str:
    pdf_document = fitz.open(stream=file.file.read(), filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def clean_up_text(text: str) -> str:
    text = text.replace('\n', ' ')
    re_spaces = re.compile(r'\s+')
    lines = text.split(' ')
    cleaned_lines = []
    for line in lines:
        cleaned_line = line.strip()
        cleaned_line = re_spaces.sub(' ', cleaned_line)
        if cleaned_line:
            cleaned_lines.append(cleaned_line)

    return ' '.join(cleaned_lines)

def format_time(elapsed_time):
    if elapsed_time < 1:
        return f"{elapsed_time * 1000:.2f} ms"
    else:
        return f"{elapsed_time:.2f} seconds"

def chunk_text(text: str, max_tokens: int = 512, overlap_percent: float = 10) -> List[str]:
    text_chunker = TextChunker(max_tokens, tokens=True, overlap_percent=overlap_percent, 
        token_estimator=OpenAITokenEstimator(), split_strategies=[sent_tokenize])
    chunks = text_chunker.chunk(text)
    for i in range(len(chunks)):
        chunks[i] = clean_up_text(chunks[i])
    return chunks

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6), retry=retry_if_not_exception_type(BadRequestError))
def get_embeddings_batch(texts: List[str], model="text-embedding-3-small") -> List[List[float]]:
    response = client.embeddings.create(input=texts, model=model, dimensions=512)
    embeddings = [item.embedding for item in response.data]
    return embeddings

def upsert_to_vectordb(chunks: List[Chunk], namespace: str):
    ns = tpuf.Namespace(namespace)
   
    ids = [chunk.vectorID for chunk in chunks]
    vectors = [chunk.embedding for chunk in chunks]
    attributes = {
        "text": [chunk.text for chunk in chunks],
        "fileID": [chunk.fileID for chunk in chunks]
    }

    ns.upsert(
        ids=ids,
        vectors=vectors,
        attributes=attributes,  # Arbitrary attributes, as long as they have consistent types
        distance_metric='cosine_distance',
    )

def run_kmeans_clustering(embeddings: List[List[float]], n_clusters: int) -> KMeans:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(embeddings)
    return kmeans

def get_centermost_chunks(kmeans: KMeans, chunks: List[Chunk]) -> List[Dict]:
    cluster_centers = kmeans.cluster_centers_
    cluster_summary_chunks = []
    embeddings = [chunk.embedding for chunk in chunks]

    # Find the centermost chunk for each cluster
    for center in cluster_centers:
        distances = [np.linalg.norm(np.array(center) - np.array(embed)) for embed in embeddings]
        centermost_idx = np.argmin(distances)
        cluster_summary_chunks.append({
            "index": centermost_idx,
            "chunk": chunks[centermost_idx].text,
            "vector_id": chunks[centermost_idx].vectorID
        })

    # Sort the cluster summary chunks based on their original indices
    cluster_summary_chunks_sorted = sorted(cluster_summary_chunks, key=lambda x: x["index"])

    # Remove the index from the final output
    final_summary_chunks = [{"chunk": chunk["chunk"], "vector_id": chunk["vector_id"]} for chunk in cluster_summary_chunks_sorted]

    return final_summary_chunks


async def generate_summary(cluster_summary_chunks: List[Dict]) -> str:
    # Prepare the input text for the GPT-3.5 Turbo model
    combined_chunks = "\n\n".join([f"Chunk {i+1}: {chunk['chunk']}" for i, chunk in enumerate(cluster_summary_chunks)])
    instructions = "You are given a sequence of key topic snippets from a larger document."
    instructions += "Please write a paragraph summary of the whole document based on the snippets given."
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": f"Please summarize this document in a concise paragraph:\n\n{combined_chunks}"}
    ]
    
    # Call the OpenAI API to generate a summary
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=3000, # Adjust as necessary
        temperature=0.7 # Adjust as necessary
    )
    
    summary = response.choices[0].message.content.strip()
    return summary


@router.post(
    "/upload",
    summary="Upload a file for processing",
    description="""
    Upload a PDF file for text extraction and summarization.
    The file will be processed to extract text, chunked into smaller parts,
    and then summarized using OpenAI's GPT-3.5 Turbo model.
    """,
    response_description="The summary of the document",
    responses={
        200: {
            "description": "Successful upload and processing",
            "content": {
                "application/json": {
                    "example": {
                        "summary": "This is a summary of the uploaded document."
                    }
                }
            },
        },
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
async def upload_file(file: UploadFile = File(...), namespace: str = Form(...)):
    s = time.time()
    print(f"[Upload] Received file {file.filename} for processing.")
    # Step 1: Extract all text from file via PyMuPDF (or other library for other file types)
    text = extract_text_from_pdf(file)
    print(f"[Extraction] Extracted text from file {file.filename} with {openai_token_estimator.estimate_tokens(text)} tokens.")

    # Step 1.5 (Optional): Generate unique hash for file to use as fileID
    # You will probably have your own way of managing fileIDs
    file_id = str(hash(text))
    
    # Step 2: Chunk text up to 512 tokens without splitting sentences (naive implemenation for now)
    # I'm using chunkipy for this demo, which by default uses stanza (semantic models) to split sentences meaninfully. 
    # Because it uses a model to chunk, it takes some time to run. So instead I set a custom split strategy using nltk's sent_tokenize.
    # You can implement your own function without relying on chunkipy in the future.
    start_time = time.time()
    chunks = chunk_text(text)
    print(f"[Chunking] Extracted {len(chunks)} chunks from file {file.filename}")
    print(f"[Chunking] Random chunk text: {chunks[np.random.randint(0, len(chunks))]}")
    print(f"[Chunking] Chunking latency: {format_time(time.time() - start_time)}")
    
    # Step 3: Convert each chunk to a vector embedding via OpenAI text-embedding-3-small (no error handling for now)
    # We use batch processing to significantly reduce the number of API calls and increase speed
    start_time = time.time()
    BATCH_SIZE = 100
    chunk_embeddings: List[Chunk] = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        embeddings = get_embeddings_batch(batch_chunks)
        for j, embedding in enumerate(embeddings):
            vector_id = f"{file_id}-{i + j}"
            chunk_embeddings.append(Chunk(text=batch_chunks[j], embedding=embedding, vectorID=vector_id, fileID=file_id))
    print(f"[Vectorization] Getting Embeddings latency: {format_time(time.time() - start_time)}")

    
    # Step 4: Upsert vectors + chunks to vectordb namespace w/ unique vectorIDs
    # I'm using turbopuffer, it's really nice. But you can use any vectorDB. All the concepts are the same.
    start_time = time.time()
    upsert_to_vectordb(chunk_embeddings, namespace)
    print(f"[Upsert] Upserting {len(chunk_embeddings)} embeddings to turbopuffer latency: {format_time(time.time() - start_time)}")
    
    # Step 5: Run KMeans clustering on chunks to identify key topics
    # Considering our maximum chunk size is 512 tokens, 
    # we want to use ~18 clusters * 512 = 9216 tokens to fit in 16K token window (with ~6.5k left for completion)
    start_time = time.time()
    kmeans = run_kmeans_clustering([chunk.embedding for chunk in chunk_embeddings], n_clusters=18)
    print(f"[Clustering] Running KMeans clustering latency: {format_time(time.time() - start_time)}")
    
    # Step 6: Sample centermost chunk from each cluster (average cluster meaning) to create an array of cluster summary chunks + hold on to their vectorIDs
    # You could sample more than one chunk per cluster, but we'll stick to one for now
    # Cluster Summary Chunk Vector IDs can be stored in a relational database
    # and used to retrieve the full text of the clustered summary chunks
    start_time = time.time()
    cluster_summary_chunks = get_centermost_chunks(kmeans, chunk_embeddings)
    print(f"[Cluster Summary] Generating cluster summary chunks latency: {format_time(time.time() - start_time)}")

    # Step 7 (Optional): Use gpt-3.5 turbo to generate a comprehensive summary of the cluster summary chunks
    start_time = time.time()
    summary = await generate_summary(cluster_summary_chunks)
    print(f"[Summary] Generated summary:\n{summary}")
    print(f"[Summary] Generating summary latency: {format_time(time.time() - start_time)}")
    print(f"[Total Latency] Upload handler total latency: {format_time(time.time() - s)}")
    
    # Step 8: Return generated summary
    return {
        "summary": summary,
    }
