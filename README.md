# prerequisites:
python3 --version

brew install python

python3 -m venv fastapi-env

# install dependencies
source fastapi-env/bin/activate

pip3 install -r requirements.txt

# start server
uvicorn main:app --reload

# testing out uploads ([/routers/retrieve.py](/routers/retrieve.py))

Go here: http://127.0.0.1:8000/docs#/default/upload_file_upload_post

Upload a file + specify a namespace

<img width="1182" alt="Screenshot 2024-07-08 at 4 05 50 PM" src="https://github.com/pashpashpash/python-rag-scaffold/assets/20898225/0e1477db-4ca0-4e88-a93a-bb38facc3225">

# testing out retrievals ([/routers/upload.py](/routers/upload.py))

Go here: http://127.0.0.1:8000/docs#/retrieve/get_context_retrieve_post

<img width="990" alt="Screenshot 2024-07-08 at 5 23 00 PM" src="https://github.com/pashpashpash/python-rag-scaffold/assets/20898225/e279e974-62a0-4ca6-8154-26fe4f674e73">

# Under the hood

## Upload API

1. Extract all text from file via PyMuPDF (or other library for other file types)
2. Chunk text up to 512 tokens without splitting sentences 
3. Convert each chunk to a vector embedding via OpenAI text-embedding-3-small 
4. Upsert vectors + chunks to vectordb namespace w/ unique vectorIDs
5. Run KMeans clustering on chunks to identify key topics
6. Sample centermost chunk from each cluster (average cluster meaning) to create an array of cluster summary chunks + store to their vectorIDs
7. Use gpt-3.5 turbo to generate a comprehensive summary of the cluster summary chunks

## Retrieval API

1. Convert query to embedding
2. Get top 10 relevant chunks via vectordb + store cosine similarity scores
3. Rerank chunks + store reranker score