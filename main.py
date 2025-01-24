# Copyright 2024
# Directory: fastapi-gcp-pro/main.py

from fastapi import FastAPI, HTTPException, Query
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



from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from uuid import uuid4
import os
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

class PDFAnalysis(BaseModel):
    file_url: str
    file_id: str

# Define output structure
class DescriptionOfPdf(BaseModel):
    title: str = Field(description="Title or main topic of the document")
    summary: str = Field(description="Brief summary of key content")
    document_type: str = Field(description="Type of document (research paper, report, etc.)")


@app.post("/analyze_pdf_with_semantic_chunking")
async def analyze_pdf_with_semantic_chunking(request: PDFAnalysis):
    """
    Chunk and embed a pdf file uploaded to supabase storage. 
    Then save the embeddings to the xx_pdf_vecs table
    The process is streamed to avoid memory issues.
    Reference:
    https://colab.research.google.com/drive/1HIf0OGr9gAgFcs4rnmgl-btuBvoU8Dm9#scrollTo=dUk-JR919ngQ

    Args:
        file_url (str): The URL of the uploaded PDF file.
        file_id (str): The ID of the uploaded filesUploaded instance.
    """

    print(f"file_url: {request.file_url}, file_id: {request.file_id}")
    async def chunk_embed_and_save():
        yield "Starting the PDF analysis...\n\n"
        loader = PyPDFLoader(
            file_path = "https://rblbfphbecdtyhzsfilu.supabase.co/storage/v1/object/public/pdfs/cm5bk7bkv000003l48u8q13nh",
            # password = "my-password",
            extract_images = True,
            # headers = None
            # extraction_mode = "plain",
            # extraction_kwargs = None,
        )

        docs = []
        docs_lazy = loader.lazy_load()
        for doc in docs_lazy:
            docs.append(doc)
        print(f"number of docs: {len(docs)}")
        yield f"chunked the pdf and got {len(docs)} docs, now starting to embed\n\n"

        text_splitter = SemanticChunker(OpenAIEmbeddings())
        new_docs = text_splitter.split_documents(docs)
        print(f"number of semantically chunked new_docs: {len(new_docs)}")  


        # file_id = "example_file_id_123"
        method="semantic_chunking"
        processed_docs = []

        for i,new_doc in enumerate(new_docs):
            updated_metadata = doc.metadata.copy()
            updated_metadata.update({
                "fileId":request.file_id,
                "method":method,
                "custom_id":str(uuid4()),
                "chunk":i
            })
            updated_doc = Document(
                page_content=new_doc.page_content,
                metadata=updated_metadata
            )
            processed_docs.append(updated_doc)
        print(f"number of processed_docs: {len(processed_docs)}")
        

        embeddings = OpenAIEmbeddings()
        supabase: Client = create_client(supabase_url, supabase_key)
        vector_store = SupabaseVectorStore(
            embedding=embeddings,
            client=supabase,
            table_name="xx_pdf_vecs",
            query_name="match_xx_pdf_vecs",
        )
        returned_docs_from_vector_store = vector_store.add_documents(processed_docs)

        print(f"number of returned_docs_from_vector_store: {len(returned_docs_from_vector_store)}")
        yield f"saved {len(returned_docs_from_vector_store)} docs to xx_pdf_vecs table, now gonna extract description.\n\n"


        # fetch the docs with the specific url performing similarity search in order to 
        # extract the description of the pdf
        docs__ = vector_store.similarity_search(
            "What is this document about? Give a brief summary or description.", 
            k=10,
            filter={"fileId":request.file_id}
        )

        print(f"the length of the returned docs: {len(docs__)}")
        llm = ChatOpenAI(model="gpt-4o-mini")
        structured_description = llm.with_structured_output(DescriptionOfPdf)
        # Create prompt template with formatting
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract key information from the provided document excerpts."),
            ("user", """Based on these document excerpts, provide a structured description:
            {context}""")
        ])

        # Create chain
        context = "\n\n".join([doc.page_content for doc in docs__])
        chain = prompt | llm.with_structured_output(DescriptionOfPdf)
        result = chain.invoke({"context": context})
        print(f"result: {result}")


        
        
        



    return StreamingResponse(chunk_embed_and_save(), media_type="text/event-stream")



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


from papers import extract_research_keywords, find_relevant_scientific_papers
from pydantic import BaseModel
from typing import List

class ResearchRequest(BaseModel):
    research_purpose: str


@app.post("/search-papers")
async def search_papers(
    request: ResearchRequest,
    limit: int = Query(default=10, le=100),
    offset: int = Query(default=0, ge=0)
):
    try:
        # Extract keywords from the research purpose
        keywords = extract_research_keywords(request.research_purpose)
        print(f"keywords: {keywords}")
        
        # Format the query for Elsevier
        formatted_query = " AND ".join(f"TITLE-ABS-KEY({keyword})" for keyword in keywords)
        
        # Get the papers
        results = find_relevant_scientific_papers(formatted_query, limit=limit, offset=offset)
        
        # Add the extracted keywords to the response
        results["extracted_keywords"] = keywords
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    





from typing import List, Optional
import httpx
from urllib.parse import quote

class PubMedArticle(BaseModel):
   pmid: str
   title: Optional[str]
   abstract: Optional[str]
   authors: List[str]
   journal: Optional[str]
   publication_date: Optional[str]

class SearchRequest(BaseModel):
   query: str
   max_results: int = 10

@app.post("/api/pubmed/search", response_model=List[PubMedArticle])
async def search_pubmed(request: SearchRequest):
   try:
       base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
       async with httpx.AsyncClient() as client:
           # Search for IDs
           search_url = f"{base_url}esearch.fcgi?db=pubmed&term={quote(request.query)}&retmax={request.max_results}&format=json"
           search_response = await client.get(search_url)
           search_data = search_response.json()
           pmids = search_data['esearchresult']['idlist']

           # Get article details
           fetch_url = f"{base_url}esummary.fcgi?db=pubmed&id={','.join(pmids)}&format=json"
           details_response = await client.get(fetch_url)
           details_data = details_response.json()

           articles = []
           for pmid in pmids:
               article_data = details_data['result'][pmid]
               articles.append(PubMedArticle(
                   pmid=pmid,
                   title=article_data.get('title', ''),
                   abstract=article_data.get('abstract', ''),
                   authors=[author['name'] for author in article_data.get('authors', [])],
                   journal=article_data.get('fulljournalname', ''),
                   publication_date=article_data.get('pubdate', '')
               ))

           return articles

   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))