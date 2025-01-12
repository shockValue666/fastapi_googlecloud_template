# Python backend (on google cloud) for Nextjs apps

## Problem statement 
    - People don't know how to launch backend

## Product requirements
    - launch a backend with Python FastAPI
    - Deploy it to Google Cloud Platform
    - Use it Nextjs frontend

# Tools
    - Fast API
    - Google Cloud
    - Cursor? 
    - Nextjs




# Tutorial

We will create a file to store the server. This file is gonna be called fastapi-gcp-pro. There we will create a fast api with some function to greet the user and return the weather. We will also deploy it to docker before deploying to google cloud platoform so we will need .dockerignore .Dockerfile .gcloudignore files. 

1. We will create the virtual environment. 
```
python3 -m venv venv__fastapi_gcp_pro
```
2. activate the virtual environment
```
source ./venv__fastapi_gcp_pro/bin/activate
```
3. create requirements.txt

4. Install the requirements
```
pip install -r requirements.txt
```

5. create main.py

6. create Dockerfile

7. create .gcloudignore

8. create .dockerignore

9. create .gitignore

10. run the application
```
uvicorn main:app --reload
```
or to match it with the docker file

```
 uvicorn main:app --reload --port 8080
```


