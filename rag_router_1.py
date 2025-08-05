import os
import json
import shutil
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from colorama import Fore, Style, init
import numpy as np

class RAGRouter:
    """
    A RAG-based router that uses ChromaDB and sentence transformers 
    to classify user prompts into appropriate routes based on semantic similarity.
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 confidence_threshold: float = 0.1,
                 max_routes: int = 3,
                 collection_name: str = "route_collection"):
        """
        Initialize the RAG router.
        
        Args:
            model_name: Sentence transformer model to use for embeddings
            confidence_threshold: Minimum confidence score for route selection
            max_routes: Maximum number of routes to return
            collection_name: Name of the ChromaDB collection
        """
        self.confidence_threshold = confidence_threshold
        self.max_routes = max_routes
        self.collection_name = collection_name
        
        # Initialize colorama
        init()
        self.clear_chroma_db()

        print(f"{Fore.YELLOW}Initializing RAG Router...{Style.RESET_ALL}")
        
        # Initialize embedding model
        print(f"{Fore.CYAN}Loading embedding model: {model_name}...{Style.RESET_ALL}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        print(f"{Fore.CYAN}Initializing ChromaDB...{Style.RESET_ALL}")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection
#        try:
#            self.collection = self.chroma_client.get_collection(name=collection_name)
#            print(f"{Fore.GREEN}Loaded existing route collection with {self.collection.count()} routes{Style.RESET_ALL}")
#        except:
        self.collection = self.chroma_client.create_collection(name=collection_name)
        print(f"{Fore.GREEN}Created new route collection{Style.RESET_ALL}")
        self._initialize_default_routes()
    
    def _initialize_default_routes(self):
        """Initialize the collection with default routes."""
        print(f"{Fore.CYAN}Adding default routes to vector database...{Style.RESET_ALL}")
        
        default_routes = [
                {
                    "route": "vector_search",
                    # BEFORE:
                    # "description": "Searches the RAG database for relevant tech support manuals, repair guides, maintenance notes, etc. Typically used for 'who, what, when, where' queries.",
                    
                    # AFTER (more specific and inclusive of 'how-to'):
                    "description": "Use for all informational queries. This includes 'how-to' questions, requests for instructions, guides, documentation, troubleshooting steps, and explanations of features or processes.",
                    "available": True,
                    "examples": [
                        "how to add a new user?",
                        "how do I reset my password?",
                        "What are the steps to troubleshoot a network issue?",
                        "How do I configure the VPN settings?",
                        "instructions for installing the software",
                        "guide to setting up a new account",
                        "can you explain the billing process?",
                        "What is is the admin password or Veevo?",
                        "What are the common error codes and their meanings?",
                        "tell me more about the advanced features",
                        "what is a sedan?",
                        "who is the CEO?",
                        "what are the store hours?",
                        "tell me about the warranty policy",
                        "explain the difference between model A and B",                        
                        "how far apart can my chargers be mounted?",
                    ]
                },
                {
                    "route": "sql_search",
                    "description": "Queries a SQL database for structured data. Specific tables are aircraft, users, sales, and support tickets. Use this for queries that require structured data retrieval.",
                    "available": True,
                    "examples": [
                        "How many users are on the platform?",
                        "What is the total sales revenue for last quarter?",
                        "List all active users in the system.",
                    ]
                },                
                {
                    "route": "web_search",
                    "description": "Conducts a general web search for topics not covered by a vector search. Use this for current events, general knowledge, or comparing the Corvette to other vehicles.",
                    "available": True,
                    "examples": [
                        "Who won the baseball game last night?",
                        "What is the current weather forecast?",
                        "Search the internet for reviews of the 2025 Corvette.",
                        "What are the latest news headlines?",
                    ]
                },
                {
                    "route": "compose_email",
                    "description": "Use this when the query explicitly asks to send an email.",
                    "available": True,
                    "examples": [
                        "Send an email to my sales representative.",
                        "Compose an email with the search results.",
                        "Email this information to me.",
                        "I'd like to send an email.",
                        "Open a new email draft."
                    ]
                },
                {
                    "route": "send_sms",
                    "description": "Sends a text message (SMS) to a specified phone number. Triggered by requests to 'text' or 'SMS'.",
                    "available": True,
                    "examples": [
                        "Text the summary to my phone.",
                        "Send me a text with the car's price.",
                        "Can you SMS that to me?",
                        "Text this to 555-123-4567.",
                        "Send a text message about the Corvette."
                    ]
                },
                {
                    "route": "create_support_ticket",
                    # BEFORE:
                    # "description": "Generates a new support ticket to report an issue, file a problem, or request assistance from the support team.",
                    
                    # AFTER (focused on intervention, not information):
                    "description": "Use when a user needs to report a system failure, an error, or a problem that requires intervention from a support agent. This is for when something is broken, not for when a user needs instructions.",
                    "available": True,
                    "examples": [
                        # These examples are already good, as they imply a problem.
                        "I need to report a problem with ...",
                        "Create a support ticket for a billing issue.",
                        "File a support request about a technical glitch.",
                        "I want to open a ticket; the website is down.",
                        "There's an issue with my account, please create a ticket."
                    ]
                },
                {
                    "route": "feedback",
                    "description": "Captures and submits user feedback, opinions, or suggestions for improvement.",
                    "available": True,
                    "examples": [
                        "I have some feedback about the search functionality.",
                        "Here's my opinion on the new Corvette design.",
                        "I have a suggestion for the mobile app.",
                        "My feedback is that the process was very smooth.",
                        "I think you could improve the documentation."
                    ]
                },
                {
                    "route": "greeting",
                    "description": "Responds to simple hellos, goodbyes, and other social niceties like 'thank you'. This route should NOT be used for questions seeking information, even if they are short or conversational.",
                    "available": True,
                    "examples": [
                        "Hello",
                        "Hi, how are you?",
                        "Good morning",
                        "Hey there",
                        "Goodbye, thank you for your help.",
                        "That's all for now, bye.",
                    ]
                },
            ]

        # Add each route to the collection
        for route_data in default_routes:
            self.add_route(
                route_name=route_data["route"],
                description=route_data["description"],
                available=route_data["available"],
                examples=route_data["examples"]
            )
        
        print(f"{Fore.GREEN}Added {len(default_routes)} default routes to vector database{Style.RESET_ALL}")
    
    def clear_chroma_db(self):
        """Clear the ChromaDB collection."""
        #delete files in chroma_db directory
        chroma_db_path = "./chroma_db"
        for filename in os.listdir(chroma_db_path):
            file_path = os.path.join(chroma_db_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        # delete the folders
        for foldername in os.listdir(chroma_db_path):
            folder_path = os.path.join(chroma_db_path, foldername)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)  
        print(f"{Fore.GREEN}Deleted all files in {chroma_db_path}{Style.RESET_ALL}")

    def add_route(self, 
                  route_name: str, 
                  description: str, 
                  available: bool = True,
                  examples: List[str] = None):
        """
        Add a new route to the vector database.
        
        Args:
            route_name: Name of the route
            description: Description of when to use this route
            available: Whether the route is currently available
            examples: List of example phrases that should trigger this route
        """
        if examples is None:
            examples = []
        
        # Create a comprehensive document for embedding
        # Include description and examples for better semantic matching
        document_text = f"{description}. Examples: {' '.join(examples)}"
        
        # Generate unique ID
        route_id = f"route_{route_name}"
        
        # Add to ChromaDB collection
        self.collection.add(
            documents=[document_text],
            metadatas=[{
                "route_name": route_name,
                "description": description,
                "available": available,
                "examples": json.dumps(examples)
            }],
            ids=[route_id]
        )
        
        print(f"{Fore.GREEN}Added route: {route_name} ({'available' if available else 'unavailable'}){Style.RESET_ALL}")

    def get_stats(self) -> dict:
        """Get statistics about the route database."""
        total_routes = self.collection.count()
        routes = self.list_routes()
        available_routes = sum(1 for route in routes if route['available'])
        return {
            "total_routes": total_routes,
            "available_routes": available_routes,
            "unavailable_routes": total_routes - available_routes,
            "confidence_threshold": getattr(self, "confidence_threshold", None)
        }

    def list_routes(self):
        """Return a list of all routes with their metadata."""
        routes = []
        results = self.collection.get(include=['metadatas'])
        for metadata in results['metadatas']:
            routes.append({
                "route_name": metadata.get("route_name") or metadata.get("route"),
                "description": metadata.get("description", ""),
                "available": metadata.get("available", True),
                "examples": metadata.get("examples", "[]"),
            })
        return routes

    def classify(self, user_prompt: str) -> Dict[str, Any]:
        """
        Classify a user prompt into the top three routes.
        Args:
            user_prompt: The user's input message
        Returns:
            Dictionary containing routes, confidence scores, and explanations
        """
        print(f"{Fore.CYAN}Classifying prompt: '{user_prompt}'{Style.RESET_ALL}")
        
        # Query the vector database
        results = self.collection.query(
            query_texts=[user_prompt],
            n_results=min(self.max_routes, self.collection.count()),  # Only get top N
            include=['documents', 'metadatas', 'distances']
        )
        
        # Always select the top N routes, regardless of confidence
        selected_routes = []
        explanations = []
        max_confidence = 0.0

        if results['ids'] and len(results['ids'][0]) > 0:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                if distance == 0:
                    confidence = 1.0
                else:
                    confidence = 1.0 / (1.0 + distance)
                
                route_name = metadata['route_name']
                available = metadata['available']
                description = metadata['description']
                
                print(f"{Fore.YELLOW}  Route: {route_name}, Confidence: {confidence:.3f}, Available: {available}{Style.RESET_ALL}")
                
                selected_routes.append(route_name)
                explanations.append(f"{route_name} (confidence: {confidence:.3f})")
                max_confidence = max(max_confidence, confidence)
                
                # Stop if we have enough routes
                if len(selected_routes) >= self.max_routes:
                    break

        # If no routes found, use 'unsupported_request'
        if not selected_routes:
            selected_routes = ["unsupported_request"]
            explanations = ["No suitable available routes found - routing to unsupported_request"]
            max_confidence = 0.5  # Default confidence for sorry route

        # Create explanation
        explanation = f"Top {self.max_routes} routes based on semantic similarity: {', '.join(explanations)}"
        
        # Generate augmented prompt for rag_search routes
        augmented_prompt = "NA"
        if "rag_search" in selected_routes:
            augmented_prompt = f"Search for comprehensive information about: {user_prompt}"
        
        result = {
            "routes": selected_routes,
            "augmented_prompt": augmented_prompt,
            "message": user_prompt,
            "confidence_score": max_confidence,
            "explanation": explanation
        }
        
        print(f"{Fore.GREEN}Classification result: {selected_routes} (confidence: {max_confidence:.3f}){Style.RESET_ALL}")

        return result
        
        def list_routes(self) -> List[Dict[str, Any]]:
            """List all routes in the database."""
            results = self.collection.get(include=['metadatas'])
            routes = []
            
            for metadata in results['metadatas']:
                routes.append({
                    "route_name": metadata['route_name'],
                    "description": metadata['description'],
                    "available": metadata['available'],
                    "examples": json.loads(metadata['examples'])
                })
            
            return routes
        
        def update_route_availability(self, route_name: str, available: bool):
            """Update the availability status of a route."""
            route_id = f"route_{route_name}"
            
            # Get current metadata
            result = self.collection.get(ids=[route_id], include=['metadatas'])
            if result['ids']:
                metadata = result['metadatas'][0]
                metadata['available'] = available
                
                # Update the collection
                self.collection.update(
                    ids=[route_id],
                    metadatas=[metadata]
                )
                
                print(f"{Fore.GREEN}Updated {route_name} availability to: {available}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Route {route_name} not found{Style.RESET_ALL}")
        
        def get_stats(self) -> Dict[str, Any]:
            """Get statistics about the route database."""
            total_routes = self.collection.count()
            routes = self.list_routes()
            available_routes = sum(1 for route in routes if route['available'])
            
            return {
                "total_routes": total_routes,
                "available_routes": available_routes,
                "unavailable_routes": total_routes - available_routes,
                "confidence_threshold": self.confidence_threshold
            }

def main():
    """Test the RAG router with sample prompts."""
    print(f"{Fore.MAGENTA}=== RAG Router Test ==={Style.RESET_ALL}")
    
    # Initialize router
    router = RAGRouter()
    
    # Display stats
    stats = router.get_stats()
    print(f"\n{Fore.CYAN}Router Stats:{Style.RESET_ALL}")
    print(f"  Total routes: {stats['total_routes']}")
    print(f"  Available routes: {stats['available_routes']}")
    print(f"  Confidence threshold: {stats['confidence_threshold']}")
    
    # Test prompts
    test_prompts = [
        "what are the motor options and specs available for the 2025 Corvette",
        "hello there",
        "send me an email with the details",
        "goodbye",
        "create a support ticket for this issue",
        "search for the word 'engine' in the document",
        "I have some feedback about the service"
    ]
    
    print(f"\n{Fore.MAGENTA}=== Testing Sample Prompts ==={Style.RESET_ALL}")
    
    for prompt in test_prompts:
        print(f"\n{Fore.YELLOW}Testing: '{prompt}'{Style.RESET_ALL}")
        result = router.classify(prompt)
        print(f"  Routes: {result['routes']}")
        print(f"  Confidence: {result['confidence_score']:.3f}")
        print(f"  Explanation: {result['explanation']}")
    
    # Interactive mode
    print(f"\n{Fore.MAGENTA}=== Interactive Mode ==={Style.RESET_ALL}")
    print(f"{Fore.GREEN}Enter prompts to classify (type 'q' to quit):{Style.RESET_ALL}")
    
    while True:
        user_input = input(f"\n{Fore.CYAN}Enter prompt: {Style.RESET_ALL}")
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
        
        if user_input.strip():
            result = router.classify(user_input)
            print(f"\n{Fore.YELLOW}Routes:{Fore.CYAN} {result['routes']}")
            print(f"{Fore.YELLOW}Message:{Fore.CYAN} {result['message']}")
            print(f"{Fore.YELLOW}Confidence Score:{Fore.CYAN} {result['confidence_score']:.3f}")
            print(f"{Fore.YELLOW}Explanation:{Fore.CYAN} {result['explanation']}")
            print(f"{Fore.YELLOW}Augmented Prompt:{Fore.CYAN} {result['augmented_prompt']}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
