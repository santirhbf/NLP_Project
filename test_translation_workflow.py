# Create test_translation_workflow.py
def test_complete_workflow():
    """Test the complete translation workflow"""
    from translator import TranslationService
    from rag_engine import RAGEngine
    
    service = TranslationService()
    
    # Test queries in different languages
    test_cases = [
        ("Qu'est-ce que le droit de propriété?", "France"),
        ("Was ist Eigentumsrecht?", "Germany"), 
        ("What is property law?", None),  # Should auto-detect
    ]
    
    for query, expected_country in test_cases:
        result = service.translate_query_and_get_country(query, expected_country)
        print(f"Query: {query}")
        print(f"Detected/Forced Country: {result['target_country']}")
        print(f"Translation needed: {result['translation_needed']}")
        print("---")