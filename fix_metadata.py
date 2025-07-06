#!/usr/bin/env python3
"""
Fix metadata handling for ChromaDB compatibility
This script ensures all metadata values are simple types (str, int, float, bool, None)
"""

def fix_metadata_for_chromadb(metadata):
    """Convert metadata to ChromaDB-compatible types."""
    fixed_metadata = {}
    
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            # Already compatible
            fixed_metadata[key] = value
        elif isinstance(value, (list, tuple)):
            # Convert to string
            fixed_metadata[key] = str(value)
        elif isinstance(value, dict):
            # Convert to string
            fixed_metadata[key] = str(value)
        else:
            # Convert to string
            fixed_metadata[key] = str(value)
    
    return fixed_metadata

def process_documents_with_fixed_metadata(documents):
    """Process documents and fix their metadata."""
    fixed_documents = []
    
    for doc in documents:
        fixed_doc = doc.copy()
        if 'metadata' in fixed_doc:
            fixed_doc['metadata'] = fix_metadata_for_chromadb(fixed_doc['metadata'])
        fixed_documents.append(fixed_doc)
    
    return fixed_documents

# Example usage:
if __name__ == "__main__":
    # Test the function
    test_metadata = {
        'file_path': '/path/to/file.pdf',
        'file_name': 'document.pdf',
        'file_type': 'pdf_text',
        'chunk_index': 1,
        'total_chunks': 5,
        'nested_dict': {'pages': 10, 'title': 'Test'},
        'list_data': ['item1', 'item2']
    }
    
    fixed = fix_metadata_for_chromadb(test_metadata)
    print("Original:", test_metadata)
    print("Fixed:", fixed) 