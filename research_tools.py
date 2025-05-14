import requests
import os
from dotenv import load_dotenv
import json
import traceback

# Semantic Scholar API endpoint
S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

def _construct_s2_api_params(
    topic: str,
    year: int = None,
    year_filter: str = None,
    min_citations: int = None
) -> dict:
    params = {
        'query': topic,
        'fields': 'paperId,title,authors,year,citationCount,url,externalIds',
    }
    if year is not None and year_filter is not None:
        if year_filter == "in":
            params['year'] = str(year)
        elif year_filter == "before":
            params['year'] = f"-{year - 1}"
        elif year_filter == "after":
            params['year'] = f"{year + 1}-"
        else:
            print(f"Warning: Invalid year_filter '{year_filter}'. Year filtering will be skipped.")
    elif year is not None and year_filter is None:
        params['year'] = str(year)

    if min_citations is not None and min_citations >= 0:
        params['minCitationCount'] = min_citations
    return params

def _format_paper_details(paper_s2_format: dict) -> dict | None:
    paper_title = paper_s2_format.get('title')
    if not paper_title:
        return None # Skip papers without titles

    return {
        "paperId": paper_s2_format.get('paperId'),
        "title": paper_title,
        "authors": ", ".join([author['name'] for author in paper_s2_format.get('authors', []) if author.get('name')]),
        "year": paper_s2_format.get('year'),
        "citationCount": paper_s2_format.get('citationCount', 0),
        "url": paper_s2_format.get('url'),
        "doi": paper_s2_format.get('externalIds', {}).get('DOI')
    }

def _handle_request_errors(e: requests.exceptions.RequestException, response_obj=None) -> str:
    if isinstance(e, requests.exceptions.HTTPError):
        error_content = "Could not retrieve error content from response."
        try:
            if response_obj and hasattr(response_obj, 'text'):
                error_content = response_obj.text
        except Exception:
            pass
        return json.dumps({"error": f"HTTP error occurred: {e}", "details": error_content})
    elif isinstance(e, requests.exceptions.ConnectionError):
        return json.dumps({"error": f"Connection error occurred: {e}"})
    elif isinstance(e, requests.exceptions.Timeout):
        return json.dumps({"error": f"Timeout error occurred: {e}"})
    else: 
        return json.dumps({"error": f"An unexpected error occurred with the request: {e}"})

def search_research_papers(
    topic: str,
    year: int = None,
    year_filter: str = None,
    min_citations: int = None,
    limit: int = 5
) -> str:
    if not topic:
        return json.dumps({"error": "Topic cannot be empty."})

    headers = {}
    # if SEMANTIC_SCHOLAR_API_KEY: # uncomment if i have an API key
    #     headers['x-api-key'] = SEMANTIC_SCHOLAR_API_KEY

    api_params = _construct_s2_api_params(topic, year, year_filter, min_citations)
    print(f"🔍 Searching Semantic Scholar (bulk) with params: {api_params} and headers: {headers}")

    all_found_papers = []
    next_token = None
    current_response = None

    try:
        while len(all_found_papers) < limit:
            current_api_params = api_params.copy()
            if next_token:
                current_api_params['token'] = next_token
            else:
                current_api_params.pop('token', None)

            current_response = requests.get(S2_API_URL, headers=headers, params=current_api_params, timeout=20)
            current_response.raise_for_status() 
            data = current_response.json()

            if 'data' in data and data['data']:
                for paper_s2_format in data['data']:
                    formatted_paper = _format_paper_details(paper_s2_format)
                    if formatted_paper:
                        all_found_papers.append(formatted_paper)
                    if len(all_found_papers) >= limit:
                        break

            if 'token' in data and data['token'] and len(all_found_papers) < limit:
                next_token = data['token']
            else:
                break

        if not all_found_papers:
            return json.dumps({"message": "No papers found matching your criteria."})
        return json.dumps(all_found_papers[:limit], indent=2)

    except requests.exceptions.RequestException as req_err:
        return _handle_request_errors(req_err, current_response)
    except Exception as e: 
        return json.dumps({"error": f"An unexpected programming error occurred: {str(e)}", "trace": traceback.format_exc()})



search_research_papers_tool_schema = {
    "name": "search_research_papers",
    "description": """Searches for research papers on Semantic Scholar based on topic,
publication year (and a filter: "in", "before", "after"), and minimum citations.
Returns a JSON string containing a list of found papers or an error message.""",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The research topic or keywords to search for (e.g., 'machine learning', 'CRISPR gene editing'). This is a required field.",
            },
            "year": {
                "type": "integer",
                "description": "Optional. The target year for filtering (e.g., 2020). Used in conjunction with year_filter.",
            },
            "year_filter": {
                "type": "string",
                "description": "Optional. How to use the year: 'in' (published in the given year), 'before' (published before the given year, e.g., year=2020 means up to 2019), 'after' (published after the given year, e.g., year=2020 means from 2021 onwards). Requires 'year' to be set.",
                "enum": ["in", "before", "after"],
            },
            "min_citations": {
                "type": "integer",
                "description": "Optional. The minimum number of citations a paper should have (e.g., 100). If not provided, no citation filter is applied.",
            },
            "limit": {
                "type": "integer",
                "description": "Optional. The maximum number of papers to return. Defaults to 5 if not specified by the user, but the agent can choose a different limit if appropriate.",
                "default": 5,
            },
        },
        "required": ["topic"],
    },
}

# basic tests
if __name__ == "__main__":
    print("--- Testing search_research_papers (using /paper/search/bulk) ---")

    # Test Case 1 Basic topic search
    print("\nTest Case 1: Basic topic search (machine learning)")
    results1 = search_research_papers(topic="machine learning", limit=2)
    print(results1)

    # Test Case 2 Topic with year "in"
    print("\nTest Case 2: Topic with year 'in' (artificial intelligence, 2022)")
    results2 = search_research_papers(topic="artificial intelligence", year=2022, year_filter="in", limit=2)
    print(results2)

    # Test Case 3 Topic with year "before" and citations
    print("\nTest Case 3: Topic with year 'before' and citations (natural language processing, before 2020, min 100 citations)")
    results3 = search_research_papers(topic="natural language processing", year=2020, year_filter="before", min_citations=100, limit=2)
    print(results3)

    # Test Case 4 Topic with year "after"
    print("\nTest Case 4: Topic with year 'after' (reinforcement learning, after 2021)")
    results4 = search_research_papers(topic="reinforcement learning", year=2021, year_filter="after", limit=2)
    print(results4)

    # Test Case 5 No results expected (highly specific or non-existent)
    print("\nTest Case 5: No results (underwater basket weaving, 2050)")
    results5 = search_research_papers(topic="underwater basket weaving", year=2050, year_filter="in", limit=2)
    print(results5)

    # Test Case 6 Empty topic
    print("\nTest Case 6: Empty topic")
    results6 = search_research_papers(topic="", limit=2)
    print(results6)

    # Test Case 7 Only min_citations
    print("\nTest Case 7: Only min_citations (deep learning, min 5000 citations)")
    results7 = search_research_papers(topic="deep learning", min_citations=5000, limit=3)
    print(results7)

    # Test Case 8: More results to test pagination (if API returns many)
    #print("\nTest Case 8: More results (machine learning, limit 7)")
    #results8 = search_research_papers(topic="machine learning", limit=7)
    #print(results8)


    print("\n--- Testing Finished ---")


