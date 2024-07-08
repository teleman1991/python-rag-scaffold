# prerequisites:
python3 --version
brew install python
python3 -m venv fastapi-env

# install dependencies
source fastapi-env/bin/activate
pip3 install -r requirements.txt

# start server
uvicorn main:app --reload
