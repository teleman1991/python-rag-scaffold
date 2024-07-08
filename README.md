# prerequisites:
python3 --version

brew install python

python3 -m venv fastapi-env

# install dependencies
source fastapi-env/bin/activate

pip3 install -r requirements.txt

# start server
uvicorn main:app --reload

# testing out uploads

Go here: http://127.0.0.1:8000/docs#/default/upload_file_upload_post

Upload a file + specify a namespace

# testing out retrievals