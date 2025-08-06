import os
import json
import chromadb
from chromadb.errors import NotFoundError
import shutil
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

# Using colorama for cleaner console output
from colorama import Fore, Style, init

class RAGRouter:
    """
    A hybrid router that uses keyword matching for specific routes and
    semantic search for all other routes.
    """
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 max_routes: int = 3,
                 collection_name: str = "route_collection_v3"):
        
        self.max_routes = max_routes
        self.collection_name = collection_name
        self.routes_config = [] # Will hold the full config for all routes
        
        init(autoreset=True)
        print(f"{Fore.GREEN}✅ Initializing Hybrid RAG Router...{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✅ Loading embedding model: {model_name}...{Style.RESET_ALL}")
        self.embedding_model = SentenceTransformer(model_name)
        print(f"{Fore.GREEN}✅ Initializing ChromaDB...{Style.RESET_ALL}")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        self._reset_collection()
        self._initialize_default_routes()
    
    def _reset_collection(self):
        """Deletes the old collection if it exists and creates a new one."""
        print(f"{Fore.GREEN}✅ Resetting collection '{self.collection_name}'...{Style.RESET_ALL}")
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f"{Fore.GREEN}✅ Deleted existing collection.{Style.RESET_ALL}")
        except Exception: # Broad exception for compatibility
            print(f"{Fore.GREEN}✅ Collection did not exist, creating new one.{Style.RESET_ALL}")

        self.collection = self.chroma_client.create_collection(name=self.collection_name)
        print(f"{Fore.GREEN}✅ Successfully created new collection.{Style.RESET_ALL}")

    def _initialize_default_routes(self):
        """Initialize the collection with default routes."""
        print(f"{Fore.CYAN}✅ Adding default routes from routes.json...{Style.RESET_ALL}", end=' ')
        
        # Load routes from external JSON file
        routes_file = "routes.json"
        try:
            with open(routes_file, 'r', encoding='utf-8') as file:
                default_routes = json.load(file)
            print(f"{Fore.GREEN}Success!{Style.RESET_ALL}")
        except FileNotFoundError:
            print(f"{Fore.RED}Error: {routes_file} not found. Please ensure the file exists.{Style.RESET_ALL}")
            default_routes = []
        except json.JSONDecodeError as e:
            print(f"{Fore.RED}Error parsing {routes_file}: {e}{Style.RESET_ALL}")
            default_routes = []

        for route_config in default_routes:
            self.add_route(route_config)
        
        print(f"{Fore.GREEN}✅ Finished adding {len(default_routes)} default routes.{Style.RESET_ALL}\n")

    def add_route(self, route_config: Dict[str, Any]):
        """
        Adds a route. If semantic, adds examples to ChromaDB. 
        Stores all route configs in memory for the classifier.
        """
        route_name = route_config['route']
        self.routes_config.append(route_config)

        # Only add semantic routes to the vector database for searching
        if route_config.get('match_type', 'semantic') == 'semantic':
            description = route_config['description']
            examples = route_config.get('examples', [])
            documents_to_add = [description] + examples

            if documents_to_add:
                metadatas = [{"route_name": route_name}] * len(documents_to_add)
                ids = [f"route_{route_name}_{i}" for i in range(len(documents_to_add))]
                self.collection.add(documents=documents_to_add, metadatas=metadatas, ids=ids)
                print(f"   🔹 Added semantic route: {Fore.CYAN}{route_name}{Style.RESET_ALL} ({len(documents_to_add)} documents)")
        else:
            print(f"   🔹 Added keyword route: {Fore.CYAN}{route_name}{Style.RESET_ALL}")

    def classify(self, user_prompt: str) -> Dict[str, Any]:
        """
        Classifies a prompt using a hybrid keyword and semantic approach.
        """
        # --- Step 1: Check for Keyword Matches ---
        # This is a hard rule check that happens before any semantic search.
        keyword_matches = []
        prompt_lower = user_prompt.lower()
        for route in self.routes_config:
            if route.get('match_type') == 'keyword':
                if any(keyword in prompt_lower for keyword in route.get('keywords', [])):
                    keyword_matches.append(route['route'])
        
        # --- Step 2: Perform Semantic Search ---
        # This runs on all documents in the vector database.
        results = self.collection.query(
            query_texts=[user_prompt],
            n_results=self.max_routes * 2, # Get extra to allow for deduplication
            include=['metadatas', 'distances']
        )
        
        # --- Step 3: Combine and Rank Results ---
        unique_routes = []
        route_confidence = {}
        
        # Add keyword matches first, giving them absolute priority and confidence
        for route_name in keyword_matches:
            if route_name not in unique_routes:
                unique_routes.append(route_name)
                route_confidence[route_name] = 1.0

        # Then, add semantic matches until we reach our max_routes limit
        if results['ids'] and results['ids'][0]:
            for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                if len(unique_routes) >= self.max_routes:
                    break # Stop if we have enough routes
                
                route_name = metadata['route_name']
                if route_name not in unique_routes:
                    unique_routes.append(route_name)
                    confidence = 1.0 / (1.0 + distance)
                    route_confidence[route_name] = confidence
        
        # Handle case where no routes are found at all
        if not unique_routes:
            unique_routes = ["unsupported_request"]
            route_confidence["unsupported_request"] = 0.0

        explanation = f"   Semantic Matches: {Fore.CYAN}{list(route_confidence.keys())}{Style.RESET_ALL}\n    Keyword Matches: {Fore.CYAN}{keyword_matches}{Style.RESET_ALL}."

        result = {
            "routes": unique_routes,
            "confidence_score": route_confidence.get(unique_routes[0], 0.0),
            "explanation": explanation,
            "message": user_prompt
        }
        return result

def delete_folder_contents(folder_path: str):
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"{Fore.GREEN}✅ Deleted directory: {folder_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Error deleting directory {folder_path}: {e}{Style.RESET_ALL}")

def main():
    """Test the RAG router with sample prompts."""
    delete_folder_contents("./chroma_db")  # Clean up old data
    router = RAGRouter()
    

    while True:
        print(f'{Fore.YELLOW}Enter prompt to classify (or q to quit):', end=' ')
        user_input = input(f"{Fore.WHITE}").strip()
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
        
        if user_input.strip():
            result = router.classify(user_input)
            print(f"{Fore.YELLOW}Top Confidence:{Fore.WHITE} {result['confidence_score']:.3f}")
            print(f"{Fore.YELLOW}Classification:\n{Fore.WHITE} {result['explanation']}")
            print(f"{Fore.YELLOW}-----------------------------------------------------\n")

if __name__ == "__main__":
    print(f'{Fore.GREEN}')
    print(' _____ _____ _____     _____ _____ _____ _____ _____ _____ ')
    print('| __  |  _  |   __|___| __  |     |  |  |_   _|   __| __  |')
    print('|    -|     |  |  |___|    -|  |  |  |  | | | |   __|    -|')
    print('|__|__|__|__|_____|   |__|__|_____|_____| |_| |_____|__|__|', 'v1.0.0')
    main()
