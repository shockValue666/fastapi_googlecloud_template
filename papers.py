from typing import List, Dict, Optional
from pydantic import BaseModel
import requests
from urllib.parse import quote
from langchain_core.pydantic_v1 import Field
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import List
import os

# convert the user string to keywords
# Pydantic model for structured output
class KeywordExtraction(BaseModel):
    keywords: List[str] = Field(description="List of relevant academic keywords")
    reasoning: str = Field(description="Brief explanation of why these keywords were chosen")

def extract_research_keywords(research_purpose: str, model_name: str = "gpt-4o-mini") -> List[str]:
    """
    Extract relevant academic keywords from a research purpose description.
    
    Args:
        research_purpose (str): Description of the research purpose/question
        model_name (str): Name of the LLM model to use
    
    Returns:
        List[str]: List of extracted keywords
    """
    # Initialize the parser with our pydantic model
    parser = PydanticOutputParser(pydantic_object=KeywordExtraction)

    # Create the prompt template
    template = """
    As an academic research assistant, analyze the following research purpose and extract the most relevant academic keywords 
    that would be useful for searching in scientific databases. Focus on specific technical terms and established concepts.

    Research Purpose: {research_purpose}

    Guidelines:
    - Extract 3-5 most relevant keywords
    - Prioritize established academic terms
    - Consider both broad and specific terms
    - Include relevant methodologies if mentioned
    - Exclude common words and generic terms

    {format_instructions}
    """

    # Create the prompt
    prompt = PromptTemplate(
        template=template,
        input_variables=["research_purpose"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Initialize the LLM
    llm = ChatOpenAI(
        temperature=0,
        model_name=model_name
    )

    # Generate the keywords
    _input = prompt.format_prompt(research_purpose=research_purpose)
    output = llm.invoke(_input.to_string())
    
    # Parse the output
    parsed_output = parser.parse(output.content)
    
    return parsed_output.keywords



# find the relevant papers based on the keywords 

class Article(BaseModel):
    title: str
    authors: List[str]
    journal: str
    doi: str
    citations: int
    affiliations: List[str]
    abstract: Optional[str] = None
    keywords: Optional[List[str]] = None
    year: Optional[int] = None

def find_relevant_scientific_papers(query: str, limit: int = 10, offset: int = 0) -> Dict:
    """
    Search for scientific papers in Elsevier's Scopus database.
    
    Args:
        query (str): Formatted query string
        limit (int): Number of results to retrieve (default: 10)
        offset (int): Starting position for pagination (default: 0)
    
    Returns:
        Dict containing results and metadata
    """
    url = "https://api.elsevier.com/content/search/scopus"
    
    params = {
        "query": query,
        "count": limit,
        "start": offset,
        "sort": "-citedby-count",  # Sort by citation count descending
        "field": "title,creator,publicationName,doi,citedby-count,affiliation,description,authkeywords,year",
    }
    
    headers = {
        "X-ELS-APIKey": "f1aab3d31ca4a16923a01804b1e80c3f",
        "Accept": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("search-results", {})
        
        articles = []
        for entry in results.get("entry", []):
            article = Article(
                title=entry.get('dc:title', 'No title available'),
                authors=[entry.get('dc:creator', 'No author available')],
                journal=entry.get('prism:publicationName', 'No journal available'),
                doi=entry.get('prism:doi', 'No DOI available'),
                citations=int(entry.get('citedby-count', '0')),
                affiliations=[
                    f"{affil.get('affilname', 'Unknown')} ({affil.get('affiliation-country', 'Unknown')})"
                    for affil in entry.get('affiliation', [])
                ],
                abstract=entry.get('dc:description', None),
                keywords=entry.get('authkeywords', '').split('|') if entry.get('authkeywords') else None,
                year=int(entry.get('prism:year', '0')) if entry.get('prism:year') else None
            )
            articles.append(article.dict())

        return {
            "total_results": int(results.get("opensearch:totalResults", 0)),
            "start_index": int(results.get("opensearch:startIndex", 0)),
            "items_per_page": int(results.get("opensearch:itemsPerPage", 0)),
            "articles": articles
        }
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")
    except ValueError as e:
        raise Exception(f"Failed to parse response: {str(e)}")