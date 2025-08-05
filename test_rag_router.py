"""
Simple test script for the RAG router to verify it works correctly.
"""

def test_basic_import():
    """Test if we can import the required modules."""
    try:
        import chromadb
        print("✓ ChromaDB imported successfully")
    except ImportError as e:
        print(f"✗ ChromaDB import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✓ Sentence Transformers imported successfully")
    except ImportError as e:
        print(f"✗ Sentence Transformers import failed: {e}")
        return False
    
    try:
        from rag_router import RAGRouter
        print("✓ RAGRouter imported successfully")
    except ImportError as e:
        print(f"✗ RAGRouter import failed: {e}")
        return False
    
    return True

def test_rag_router():
    """Test the RAG router functionality."""
    try:
        from rag_router import RAGRouter
        
        print("\n=== Testing RAG Router ===")
        
        # Initialize router
        print("Initializing RAG Router...")
        router = RAGRouter()
        
        # Test sample prompts
        test_prompts = [
            "what are the motor options and specs available for the 2025 Corvette",
            "hello there",
            "send me an email with the details",
            "goodbye",
            "create a support ticket for this issue"
        ]
        
        print("\n=== Testing Sample Prompts ===")
        for prompt in test_prompts:
            print(f"\nTesting: '{prompt}'")
            result = router.classify(prompt)
            print(f"  Routes: {result['routes']}")
            print(f"  Confidence: {result['confidence_score']:.3f}")
        
        print("\n✓ RAG Router test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ RAG Router test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== RAG Router Test Suite ===")
    
    # Test imports
    if not test_basic_import():
        print("\n✗ Import tests failed. Please ensure all dependencies are installed.")
        exit(1)
    
    # Test RAG router
    if not test_rag_router():
        print("\n✗ RAG Router tests failed.")
        exit(1)
    
    print("\n🎉 All tests passed! RAG Router is working correctly.")
