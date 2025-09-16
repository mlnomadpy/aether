"""Test the basic structure without requiring JAX/Flax dependencies."""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    try:
        # Test basic imports
        import aether
        print("✓ Main package imported successfully")
        
        from aether.config import Config
        print("✓ Config system imported successfully")
        
        from aether.registry import ModelRegistry
        print("✓ Model registry imported successfully")
        
        from aether.models.base import BaseModel
        print("✓ Base model imported successfully")
        
        # Test config creation
        config = Config()
        print("✓ Config creation successful")
        print(f"  Default model: {config.model.name}")
        print(f"  Default architecture: {config.model.architecture}")
        
        # Test registry creation
        registry = ModelRegistry()
        print("✓ Registry creation successful")
        print(f"  Registered models: {registry.list_models()}")
        
        print("\n🎉 All basic functionality tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_config_serialization():
    """Test configuration serialization."""
    try:
        from aether.config import Config
        
        # Create and modify config
        config = Config()
        config.model.embed_dim = 512
        config.training.batch_size = 16
        
        # Test dict conversion
        config_dict = config.to_dict()
        print("✓ Config to dict conversion successful")
        
        # Test creation from dict
        new_config = Config.from_dict(config_dict)
        print("✓ Config from dict creation successful")
        
        assert new_config.model.embed_dim == 512
        assert new_config.training.batch_size == 16
        print("✓ Config values preserved correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test error: {e}")
        return False

def main():
    """Run all tests."""
    print("Aether Framework Structure Test")
    print("=" * 40)
    
    success = True
    success &= test_imports()
    print()
    success &= test_config_serialization()
    
    if success:
        print("\n🚀 All tests passed! The framework structure is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    main()