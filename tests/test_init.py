"""
Tests for module initialization.

Tests cover:
- Package version
- Package metadata
- Module imports
"""

import pytest
import src


class TestModuleMetadata:
    """Test module-level metadata."""
    
    def test_version_exists(self):
        """Test that __version__ is defined."""
        assert hasattr(src, '__version__')
    
    def test_version_format(self):
        """Test version follows semantic versioning."""
        assert src.__version__ == "1.0.0"
        
        # Should be in format X.Y.Z
        parts = src.__version__.split('.')
        assert len(parts) == 3
        
        # All parts should be numeric
        for part in parts:
            assert part.isdigit()
    
    def test_author_exists(self):
        """Test that __author__ is defined."""
        assert hasattr(src, '__author__')
    
    def test_author_value(self):
        """Test author metadata value."""
        assert src.__author__ == "Michael Eakins"
        assert isinstance(src.__author__, str)


class TestModuleDocstring:
    """Test module documentation."""
    
    def test_module_has_docstring(self):
        """Test that module has a docstring."""
        assert src.__doc__ is not None
    
    def test_docstring_content(self):
        """Test that docstring contains expected information."""
        assert "AI Model Monitoring" in src.__doc__
        assert "monitoring" in src.__doc__.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
