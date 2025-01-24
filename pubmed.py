import requests
from urllib.parse import quote

def search_pubmed(query, max_results=10):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    # First get the IDs of matching articles
    search_url = f"{base_url}esearch.fcgi?db=pubmed&term={quote(query)}&retmax={max_results}&format=json"
    search_response = requests.get(search_url)
    pmids = search_response.json()['esearchresult']['idlist']
    
    # Then fetch the details for each article
    fetch_url = f"{base_url}esummary.fcgi?db=pubmed&id={','.join(pmids)}&format=json"
    details_response = requests.get(fetch_url)
    
    return details_response.json()



