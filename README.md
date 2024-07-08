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
