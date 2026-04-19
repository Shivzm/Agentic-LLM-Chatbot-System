from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

@tool
def search_tool(query: str) -> str:
    """Search the web for informtion"""
    #Using Tavily for web search
    search = TavilySearchResults(max_results= 3)
    results = search.invoke(query)
    return str(results)
