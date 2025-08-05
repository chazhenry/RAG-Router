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
        print(f"{Fore.YELLOW}Initializing Hybrid RAG Router...{Style.RESET_ALL}")
        
        print(f"{Fore.CYAN}Loading embedding model: {model_name}...{Style.RESET_ALL}")
        self.embedding_model = SentenceTransformer(model_name)
        
        print(f"{Fore.CYAN}Initializing ChromaDB...{Style.RESET_ALL}")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        self._reset_collection()
        self._initialize_default_routes()
    
    def _reset_collection(self):
        """Deletes the old collection if it exists and creates a new one."""
        print(f"{Fore.CYAN}Resetting collection '{self.collection_name}'...{Style.RESET_ALL}")
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f"{Fore.GREEN}Deleted existing collection.{Style.RESET_ALL}")
        except Exception: # Broad exception for compatibility
            print(f"{Fore.YELLOW}Collection did not exist, creating new one.{Style.RESET_ALL}")
            
        self.collection = self.chroma_client.create_collection(name=self.collection_name)
        print(f"{Fore.GREEN}Successfully created new collection.{Style.RESET_ALL}")

    def _initialize_default_routes(self):
        """Initialize the collection with default routes."""
        print(f"{Fore.CYAN}Adding default routes...{Style.RESET_ALL}")
        
        # ARCHITECTURE CHANGE: Route definitions now include 'match_type' and 'keywords'
        default_routes = [
            {
                "route": "support_ticket",
                "match_type": "keyword",
                "keywords": ["snow", "servicenow", "service now", "salesforce", "support ticket", "create ticket", "log this"],
                "description": "Generates a support ticket in a specific system like ServiceNow or Salesforce.",
                "examples": [
                    "Create a ServiceNow ticket.",
                    "I need to open a SNOW ticket.",
                    "Log this in Service Now.",
                    "Make a new case in Salesforce."
                ]
            },
            {
                "route": "vector_search",
                "match_type": "semantic",
                "description": "The primary route for all informational queries. This is the default for any question structured as 'what is', 'who is', 'what are', 'explain', or 'tell me about'. Use this for 'how-to' questions, requests for instructions, guides, and documentation.",
                "examples": [
                    "what is a critical system error?",
                    "tell me about error code 503",
                    "explain what a kernel panic is",
                    "what is a sedan?", "who is the CEO?", "what are the store hours?",
                    "tell me about the warranty policy",
                    "how to add a new user?", "how do I reset my password?",
                    "What are the steps to troubleshoot a network issue?",
                    "How do I configure the VPN settings?",
                ]
            },
            {
                "route": "parts_search",
                "match_type": "semantic",
                "description": "use when the user is looking for specific parts, components, or products. This is for queries that ask for a part number, product name, or specific item.",
                "examples": [
                    "How many users are on the platform?",
                    "What is the total sales revenue for last quarter?",
                ]
            },            
            {
                "route": "sql_search",
                "match_type": "semantic",
                "description": "Queries a SQL database for structured data.",
                "examples": [
                    "How many users are on the platform?",
                    "What is the total sales revenue for last quarter?",
                ]
            },
            {
                "route": "web_search",
                "match_type": "semantic", 
                "description": "Conducts a general web search. Use for current events, general knowledge, or to get the latest external information and broader public context on a technical issue. A good secondary choice after vector_search.",
                "examples": [
                    "What is a 'kernel panic'?",
                    "Find articles comparing different types of computer memory.",
                    "Search for common causes of network latency.",
                    "What are the differences between SQL and NoSQL databases?",
                    "Who won the baseball game last night?",
                    "What is the current weather forecast?",
                    "Search the internet for reviews of the 2025 Corvette.",
                ]
            },
            {
                "route": "email_send",
                "match_type": "semantic",
                "description": "Use this when the query explicitly asks to send or compose an email.",
                "examples": [
                    "Send an email to my sales representative.",
                    "Email this information to me.",
                    "Send me an email to bob@gmail.com.",
                ]
            },
            {
                "route": "sms_send",
                "match_type": "semantic",
                "description": "Use this when the query explicitly asks to send and SMS text message.",
                "examples": [
                    "Send an SMS to 919-555-1212",
                    "Text this information to my phone.",
                    "Shoot me a text with this answer.",
                    "Send me a text message.",
                ]
            },            
            {
                "route": "greeting",
                "match_type": "semantic",
                "description": "Responds to simple hellos, goodbyes, and other social niceties.",
                "examples": [ "Hello", "Hi, how are you?", "Good morning", "Goodbye" ]
            },
            {
                "route": "clarification",
                "match_type": "semantic",
                "description": "Handles meta-questions about the conversation itself. Use when the user asks for clarification on the previous response, asks to elaborate, or uses vague phrases that reference the ongoing dialogue.",
                "examples": [
                    "what do you mean by that?",
                    "Can you explain that further?",
                    "Tell me more about that.",
                    "What does that imply?",
                    "Could you elaborate?",
                    "Go on.",
                    "And then what?",
                    "Why is that the case?",
                    "What are you referring to?",
                    "Can you say that again in Spanish?",
                    "Summarize that last answer for me.",
                    "Put that into a list.",
                    "Show me that as a table."                    
                ]
            },            
            {
                "route": "reset_context",
                "match_type": "semantic",
                "description": "Resets the conversation. Use when the user wants to start over, change topics completely, or explicitly asks to clear the context of the current dialogue.",
                "examples": [
                    "Let's start over.",
                    "Clear context.",
                    "Forget everything we just talked about.",
                    "Let's try something else.",
                    "Reset."
                ]
            },
            {
                "route": "help",
                "match_type": "semantic",
                "description": "Provides help and information about the AI's capabilities and available tools. Triggered when the user asks what the system can do or requests assistance.",
                "examples": [
                    "help",
                    "What can you do?",
                    "List all available tools.",
                    "How does this work?",
                    "What are my options?"
                ]
            },                        
        {
                "route": "feedback",
                "match_type": "semantic",
                "description": "Captures explicit user feedback, both positive and negative, about the quality of the previous response or the system's performance. This is for direct praise or correction, not for when the user is confused or asking for clarification.",
                "examples": [
                    # Negative Feedback / Correction
                    "No, that's incorrect.",
                    "That's not what I asked for.",
                    "You are wrong.",
                    "That answer wasn't helpful.",
                    "you suck",
                    # Positive Feedback / Acknowledgment
                    "That was the right answer, thank you.",
                    "Perfect, that's exactly what I needed.",
                    "Great job.",
                    "This was very helpful.",
                    # Neutral Feedback / Suggestions
                    "I have a suggestion for improvement.",
                    "This process is too slow."
                ]
            },            
        ]

        for route_config in default_routes:
            self.add_route(route_config)
        
        print(f"\n{Fore.GREEN}Finished adding {len(default_routes)} default routes.{Style.RESET_ALL}")

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
                print(f"Added semantic route: {Fore.CYAN}{route_name}{Style.RESET_ALL} ({len(documents_to_add)} documents)")
        else:
            print(f"Loaded keyword route: {Fore.CYAN}{route_name}{Style.RESET_ALL}")

    def classify(self, user_prompt: str) -> Dict[str, Any]:
        """
        Classifies a prompt using a hybrid keyword and semantic approach.
        """
        print(f"\n{Fore.MAGENTA}Classifying prompt: '{user_prompt}'{Style.RESET_ALL}")
        
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

        explanation = f"Keyword Matches: {keyword_matches}. Semantic Matches: {list(route_confidence.keys())}"
        
        result = {
            "routes": unique_routes,
            "confidence_score": route_confidence.get(unique_routes[0], 0.0),
            "explanation": explanation,
            "message": user_prompt
        }
        
        print(f"{Fore.GREEN}Classification result: {result['routes']} (Top confidence: {result['confidence_score']:.3f}){Style.RESET_ALL}")
        return result

def delete_folder_contents(folder_path: str):
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"{Fore.GREEN}Deleted directory: {folder_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error deleting directory {folder_path}: {e}{Style.RESET_ALL}")

def main():
    """Test the RAG router with sample prompts."""
    print(f"\n{Fore.BLUE}{'='*20} RAG Router Test {'='*20}{Style.RESET_ALL}")

    delete_folder_contents("./chroma_db")  # Clean up old data
    router = RAGRouter()
    
    print(f"\n{Fore.BLUE}--- Testing Sample Prompts ---{Style.RESET_ALL}")
    test_prompts = [
        "what is the black screen of death",
        "I need to create a support ticket",
        "how do I reset my password?",
        "create a support ticket for this issue",
        "hello there",
        "send me an email with the details",
        "what is the weather like today?",
        "shoot me a text with this answer",
    ]
    
    #for prompt in test_prompts:
    #    router.classify(prompt)

    print(f"\n{Fore.BLUE}--- Interactive Mode ---{Style.RESET_ALL}")
    print("Enter prompts to classify (type 'q' to quit):")
    
    while True:
        user_input = input(f"\n{Fore.CYAN}Enter prompt: {Style.RESET_ALL}")
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
        
        if user_input.strip():
            result = router.classify(user_input)
            print(f"\n{Fore.YELLOW}Routes:{Fore.WHITE} {result['routes']}")
            print(f"{Fore.YELLOW}Top Confidence:{Fore.WHITE} {result['confidence_score']:.3f}")
            print(f"{Fore.YELLOW}Explanation:{Fore.WHITE} {result['explanation']}")

if __name__ == "__main__":
    main()
