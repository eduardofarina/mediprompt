import datetime
import json
from typing import Dict, Any
from openai import OpenAI
from agno.tools.pubmed import PubmedTools
from agno.tools.arxiv import ArxivTools
import os

class LiteratureSearchAgent:
    """Agent for conducting medical literature searches and synthesizing results."""
    
    def __init__(self, prompt_library):
        self.prompt_library = prompt_library
        self.system_prompt = prompt_library.get_formatted_prompt("literature-search-base")
        self.pubmed_tools = PubmedTools()
        self.arxiv_tools = ArxivTools()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def get_llm_response(self, messages, model="gpt-4o", temperature=0.7):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error getting response from OpenAI: {str(e)}"
    
    def search_literature(self, query: str) -> Dict[str, Any]:
        """
        Search medical literature based on the query.
        Returns a dictionary with search results and synthesis.
        """
        try:
            pubmed_results = self.pubmed_tools.search_pubmed(query, max_results=5)
        except Exception as e:
            pubmed_results = f"PubMed search error: {str(e)}"
        try:
            arxiv_results = self.arxiv_tools.search_arxiv(query, max_results=5)
        except Exception as e:
            arxiv_results = f"ArXiv search error: {str(e)}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
            Please search for and synthesize literature on the following query:
            Query: {query}
            PubMed results:
            {pubmed_results}
            ArXiv results:
            {arxiv_results}
            Synthesize these results into a comprehensive but concise summary.
            Focus on the most relevant findings, the strength of evidence,
            and any consensus or controversies in the literature.
            """}
        ]
        synthesis = self.get_llm_response(messages)
        return {
            "query": query,
            "pubmed_results": pubmed_results,
            "arxiv_results": arxiv_results,
            "synthesis": synthesis,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def formulate_pico_query(self, clinical_question: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
            Please reformulate the following clinical question using the PICO framework
            (Population, Intervention, Comparison, Outcome):
            Clinical Question: {clinical_question}
            Extract the PICO elements and formulate an optimized search query for medical literature databases.
            """}
        ]
        response = self.get_llm_response(messages)
        return response 