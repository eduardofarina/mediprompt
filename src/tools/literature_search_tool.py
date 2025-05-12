from agno.tools import BaseTool
import requests
from typing import Dict, Any, Optional, List
import os
import json

class LiteratureSearchTool(BaseTool):
    """
    Tool for searching medical literature databases like PubMed.
    """
    
    name = "literature_search"
    description = "Search medical literature databases for information relevant to a clinical case"
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("PUBMED_API_KEY", "")
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _run(self, query: str, max_results: int = 5, **kwargs) -> Any:
        """
        Run the literature search with the given query.
        
        Args:
            query: Search query for medical literature
            max_results: Maximum number of results to return
            
        Returns:
            Dict containing search results
        """
        # Check cache first
        cache_key = self._generate_cache_key(query, max_results)
        cached_results = self._check_cache(cache_key)
        if cached_results:
            return cached_results
            
        # If not in cache, perform the search
        try:
            # In a real implementation, this would call the PubMed API
            # For now, we'll simulate the response
            results = self._simulate_pubmed_search(query, max_results)
            
            # Cache the results
            self._cache_results(cache_key, results)
            
            return results
        except Exception as e:
            return {
                "error": str(e),
                "query": query,
                "results": []
            }
    
    def _generate_cache_key(self, query: str, max_results: int) -> str:
        """Generate a cache key for the search."""
        return f"{query}_{max_results}".replace(" ", "_").lower()
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if results are in cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def _cache_results(self, cache_key: str, results: Dict[str, Any]):
        """Cache the search results."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        with open(cache_file, "w") as f:
            json.dump(results, f, indent=2)
    
    def _simulate_pubmed_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """
        Simulate a PubMed search for development/testing purposes.
        
        In a real implementation, this would be replaced with actual API calls.
        """
        # Simulate search results based on query terms
        simulated_results = []
        
        # Parse query terms
        terms = query.lower().split()
        
        # Common medical conditions to match
        medical_conditions = {
            "pneumonia": [
                {
                    "title": "Diagnosis and Treatment of Adults with Community-acquired Pneumonia",
                    "authors": "Metlay JP, et al.",
                    "journal": "Am J Respir Crit Care Med",
                    "year": 2019,
                    "pmid": "31573350",
                    "abstract": "This document provides evidence-based clinical practice guidelines on the management of adult patients with community-acquired pneumonia."
                },
                {
                    "title": "Antibiotic Treatment Strategies for Community-Acquired Pneumonia in Adults",
                    "authors": "Postma DF, et al.",
                    "journal": "N Engl J Med",
                    "year": 2015,
                    "pmid": "25830421",
                    "abstract": "In patients with clinically suspected community-acquired pneumonia admitted to non-intensive care unit wards, a strategy of preferred empirical treatment with beta-lactam monotherapy was noninferior to strategies with beta-lactam-macrolide combination therapy or fluoroquinolone monotherapy."
                }
            ],
            "copd": [
                {
                    "title": "Global Strategy for the Diagnosis, Management, and Prevention of Chronic Obstructive Lung Disease",
                    "authors": "GOLD Committee",
                    "journal": "Am J Respir Crit Care Med",
                    "year": 2023,
                    "pmid": "36052132",
                    "abstract": "This report presents evidence-based recommendations for the diagnosis, management, and prevention of COPD."
                },
                {
                    "title": "Pharmacologic Management of COPD: An Official ATS Clinical Practice Guideline",
                    "authors": "Nici L, et al.",
                    "journal": "Am J Respir Crit Care Med",
                    "year": 2020,
                    "pmid": "32283960",
                    "abstract": "This document provides clinical recommendations for the pharmacologic treatment of COPD."
                }
            ],
            "cardiac": [
                {
                    "title": "2020 ESC Guidelines for the management of acute coronary syndromes in patients presenting without persistent ST-segment elevation",
                    "authors": "Collet JP, et al.",
                    "journal": "Eur Heart J",
                    "year": 2021,
                    "pmid": "33085966",
                    "abstract": "These guidelines address the management of patients with suspected or confirmed acute coronary syndromes without persistent ST-segment elevation."
                },
                {
                    "title": "The HEART Pathway randomized trial: identifying emergency department patients with acute chest pain for early discharge",
                    "authors": "Mahler SA, et al.",
                    "journal": "Circ Cardiovasc Qual Outcomes",
                    "year": 2015,
                    "pmid": "25737484",
                    "abstract": "The HEART Pathway reduces objective cardiac testing during 30 days, increases early discharges, and reduces length of stay without increasing adverse events."
                }
            ]
        }
        
        # Match query terms to medical conditions
        matched_articles = []
        for term in terms:
            for condition, articles in medical_conditions.items():
                if term in condition:
                    matched_articles.extend(articles)
        
        # If no specific matches, return general medical articles
        if not matched_articles:
            matched_articles = [
                {
                    "title": "Clinical Reasoning: A Guide to Improving Teaching and Practice",
                    "authors": "Norman G.",
                    "journal": "Med Educ",
                    "year": 2018,
                    "pmid": "30265414",
                    "abstract": "This paper discusses clinical reasoning models and their application in medical education and practice."
                },
                {
                    "title": "Diagnostic Error in Medicine: Analysis of 583 Physician-Reported Errors",
                    "authors": "Schiff GD, et al.",
                    "journal": "Arch Intern Med",
                    "year": 2009,
                    "pmid": "19901140",
                    "abstract": "This study analyzes diagnostic errors and suggests approaches for their prevention."
                }
            ]
        
        # Deduplicate and limit results
        seen_pmids = set()
        for article in matched_articles:
            if article["pmid"] not in seen_pmids and len(simulated_results) < max_results:
                seen_pmids.add(article["pmid"])
                simulated_results.append(article)
        
        return {
            "query": query,
            "result_count": len(simulated_results),
            "results": simulated_results
        } 