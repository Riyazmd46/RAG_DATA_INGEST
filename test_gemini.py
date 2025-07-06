#!/usr/bin/env python3
"""
Test script to verify Gemini API connectivity and model availability
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def test_gemini_models():
    """Test different Gemini models to find the correct ones."""
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        return
    
    genai.configure(api_key=api_key)
    
    # List of models to test
    models_to_test = [
        "gemini-1.5-flash",
        "gemini-1.5-pro", 
        "gemini-pro",
        "gemini-pro-vision"
    ]
    
    print("Testing Gemini models...")
    print("=" * 50)
    
    for model_name in models_to_test:
        try:
            print(f"Testing model: {model_name}")
            model = genai.GenerativeModel(model_name)
            
            # Test text generation
            response = model.generate_content("Hello! Can you confirm this model is working?")
            print(f"  ✅ {model_name} - Text generation: SUCCESS")
            print(f"     Response: {response.text[:100]}...")
            
            # Test if it supports vision (for vision models)
            if "vision" in model_name or "1.5" in model_name:
                try:
                    # Test vision capability
                    response = model.generate_content(["Describe this text: Hello World"])
                    print(f"  ✅ {model_name} - Vision capability: SUCCESS")
                except Exception as e:
                    print(f"  ⚠️  {model_name} - Vision capability: {str(e)[:50]}...")
            
            print()
            
        except Exception as e:
            print(f"  ❌ {model_name} - ERROR: {str(e)[:100]}...")
            print()

def test_embeddings():
    """Test embedding model."""
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        api_key = os.getenv("GEMINI_API_KEY")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # Test embedding generation
        test_text = "Hello world"
        embedding = embeddings.embed_query(test_text)
        
        print("✅ Embedding model: SUCCESS")
        print(f"   Embedding length: {len(embedding)}")
        print(f"   Sample values: {embedding[:5]}")
        
    except Exception as e:
        print(f"❌ Embedding model - ERROR: {str(e)[:100]}...")

if __name__ == "__main__":
    print("Gemini API Test")
    print("=" * 50)
    
    test_gemini_models()
    print("\nTesting Embeddings:")
    print("-" * 30)
    test_embeddings() 