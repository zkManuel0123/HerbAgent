import requests
from typing import List, Dict, Optional
import json

class UMLSMapper:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://uts-ws.nlm.nih.gov"
        self.version = "current"

    def search_term(self, term: str) -> Optional[Dict]:
        """Search UMLS for a term and return the most relevant match"""
        content_endpoint = f"/rest/search/{self.version}"
        full_url = self.base_url + content_endpoint
        
        try:
            params = {
                'string': term,
                'apiKey': self.api_key,
                'pageNumber': 1  # 只需要第一页
            }
            
            print(f"Searching for term: {term}")
            response = requests.get(full_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            items = data['result']['results']
            
            # 只返回第一个匹配结果（最相关的）
            if items:
                return {
                    'name': items[0]['name'],
                    'cui': items[0]['ui']
                }
            return None
            
        except Exception as e:
            print(f"Error searching term '{term}': {str(e)}")
            return None

    def batch_process(self, terms: List[str]) -> Dict[str, Optional[Dict]]:
        """Process multiple terms and return the best match for each"""
        results = {}
        for term in terms:
            match = self.search_term(term)
            if match:
                results[term] = match
        return results

# Usage example
if __name__ == "__main__":
    API_KEY = "0de84219-a44c-4841-a285-8259e1fd0599"  # Get from https://uts.nlm.nih.gov/uts/signup-login
    
    mapper = UMLSMapper(API_KEY)
    terms = ["Thyroid Neoplasms", "Thyroid Cancer", "Thyroid carcinoma"]
    
    results = mapper.batch_process(terms)
    print(json.dumps(results, indent=2))