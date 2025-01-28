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


# Deploy to Google Cloud Platform

1. install the gcloud cli
```
brew install google-cloud-sdk
```

2. authenticate with GCP
```
# Login to google account
gcloud auth login
```

3. initialize the project in gcp
    - go to https://console.cloud.google.com/
    - top left > new project 
    - sidebar > cloud run
    - MY_PROJECT_ID is the name of the project we created in the second bullet

```
# set the project id
gcloud config set project MY_PROJECT_ID
```

4. Enable required APIs
```
# enable cloud run api
gcloud services enable run.googleapis.com

# enable container registry api
gcloud services enable containerregistry.googleapis.com
```

5. Build and push the docker image to Google Container Registry
```
# set the project id in a variable
export PROJECT_ID=$(gcloud config get-value project)

# Build the container image
gcloud builds submit --tag gcr.io/$PROJECT_ID/fastapi-gcp-nextjs

# Alternative methods
docker build -t gcr.io/$PROJECT_ID/fastapi-gcp-nextjs .
docker push gcr.io/$PROJECT_ID/fastapi-gcp-nextjs
```

6. Deploy to Cloud Run
```
gcloud run deploy fastapi-gcp-nextjs \
    --image gcr.io/$PROJECT_ID/fastapi-gcp-nextjs \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

7. Connect to github so that it redeploy on push
    - push the code to github
    - go to the cloud run instance
    - navbar > Connect Repo > SET UP WITH CLOUD BUILD > 2.Build configuration: /Dockerfile
    - create (bottom of the page)




# Directions for testing and developing 

1. first deploy the docker image locally 
```
gcloud builds submit --tag gcr.io/$PROJECT_ID/fastapi-gcp-nextjs
```

2. then in order to deploy with env variables
    a. 
    ```
    gcloud run deploy fastapi-gcp-nextjs \
        --image gcr.io/$PROJECT_ID/fastapi-gcp-nextjs \
        --set-env-vars "KEY1=value1,KEY2=value2"
    ```
    b. 
        - go to google cloud service
        - click "Edit & Deploy New Revision" 
        - under "Variables & Secrets" tab, add the env variables

# Automatic deployment from github (since i can't pass the env variables directly to the gcloud)
    1. crate cloudbuild.yaml