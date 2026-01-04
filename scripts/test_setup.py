#!/usr/bin/env python3
"""
Quick test script to verify the agentic-patterns setup.

Run with: uv run scripts/test_setup.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from agentic_patterns.common import settings, ModelFactory, create_writer
        print("✓ Common modules imported successfully")
        
        from agentic_patterns.patterns.chaining_01 import run, ChainingConfig
        print("✓ Pattern 01_chaining imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config():
    """Test environment configuration."""
    print("\nTesting configuration...")
    try:
        from agentic_patterns.common import settings
        
        if settings.OPENAI_API_KEY:
            print(f"✓ OPENAI_API_KEY found (length: {len(settings.OPENAI_API_KEY)})")
        else:
            print("⚠ OPENAI_API_KEY not set in .env")
        
        if settings.ANTHROPIC_API_KEY:
            print(f"✓ ANTHROPIC_API_KEY found (length: {len(settings.ANTHROPIC_API_KEY)})")
        else:
            print("⚠ ANTHROPIC_API_KEY not set in .env")
        
        print(f"✓ Results directory: {settings.RESULTS_DIR}")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_model_factory():
    """Test model factory."""
    print("\nTesting ModelFactory...")
    try:
        from agentic_patterns.common import ModelFactory, settings

        # Test creating models for each provider (if API keys are available)
        test_models = []

        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your_openai_api_key_here":
            test_models.append(("gpt-4", "OpenAI"))

        if settings.ANTHROPIC_API_KEY and settings.ANTHROPIC_API_KEY != "your_anthropic_api_key_here":
            test_models.append(("claude-sonnet-4-5-20250929", "Anthropic"))

        if settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY != "your_gemini_api_key_here":
            test_models.append(("gemini-pro", "Google"))

        if not test_models:
            print("⚠ No valid API keys found - skipping model creation test")
            print("✓ ModelFactory class imported successfully")
            return True

        # Try to create a model instance
        model_name, provider = test_models[0]
        model = ModelFactory.create(model_name, temperature=0.7)
        print(f"✓ ModelFactory can create {provider} models (tested: {model_name})")

        if len(test_models) > 1:
            providers = [p for _, p in test_models]
            print(f"✓ API keys available for: {', '.join(providers)}")

        return True
    except Exception as e:
        print(f"✗ ModelFactory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Agentic Patterns - Setup Test")
    print("=" * 60)
    
    results = []
    results.append(test_imports())
    results.append(test_config())
    results.append(test_model_factory())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All tests passed! Setup is ready.")
        print("\nNext steps:")
        print("1. Make sure your .env file has API keys")
        print("2. Run: uv run src/agentic_patterns/patterns/chaining_01/run.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())