# Copyright 2024
# Directory: fastapi-gcp-pro/main.py

from fastapi import FastAPI
from datetime import datetime
import requests
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from fastapi.responses import StreamingResponse

app = FastAPI(title="FastAPI GCP Pro")

# Define allowed origins for CORS
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://launch-an-app.vercel.app"
]

# Add CORS middleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,  # Allow credentials (cookies, authorization headers)
    allow_methods=["*"],    # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)


@app.get("/")
async def root():
    """Root endpoint returning a welcome message."""
    return {"message": "Welcome to FastAPI GCP Pro"}

@app.get("/greet/{name}")
async def greet(name: str):
    """
    Greet endpoint that returns a personalized greeting.
    
    Args:
        name (str): Name of the person to greet
    """
    return {"message": f"Hello, {name}! I think you are great!"}

@app.get("/weather")
async def fetch_weather_today():
    """Fetch current weather data from a mock API."""
    # Note: In a real application, you would use an actual weather API
    # This is a mock response
    mock_weather = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "temperature": 22,
        "condition": "Sunny",
        "location": "Sample City"
    }
    return mock_weather 



@app.get("/progress")
async def progress():
    """
    Endpoint that streams progress updates to the client every 10 seconds.
    """
    async def progress_stream():
        for i in range(1, 11):  # 10 updates (every 10 seconds for 100 seconds total)
            progress = i * 10  # Percentage progress
            yield f"data: Progress: {progress}%\n\n"  # SSE format
            await asyncio.sleep(10)  # Wait for 10 seconds
        yield "data: Progress: 100% - done\n\n"

    return StreamingResponse(progress_stream(), media_type="text/event-stream")




# use ngrok to expose the local server to the internet
@app.post("/register_webhook")
async def register_webhook(webhook_url: str):
    """
    Register a webhook URL to send updates.
    Args:
        webhook_url (str): The URL to call back when the task is done.
    """
    # Simulate a long task
    async def simulate_task():
        await asyncio.sleep(100)  # Simulate task duration (100 seconds)
        # Send callback to webhook
        response = requests.post(webhook_url, json={"status": "done", "message": "Task completed!"})
        print(f"Webhook sent. Status code: {response.status_code}")

    asyncio.create_task(simulate_task())  # Run task in background
    return {"message": "Webhook registered. You will be notified when the task is complete."}
