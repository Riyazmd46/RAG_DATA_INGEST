import os
import re
from typing import List, Dict, Any, Optional
import PyPDF2
import tabula
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from PIL import Image
import io

from config import Config

class TextProcessor:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.GEMINI_API_KEY
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        self.vision_model = genai.GenerativeModel(Config.GEMINI_VISION_MODEL)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )
        
    def read_text_file(self, file_path: str) -> str:
        """Read text from various text file formats."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
    
    def extract_pdf_content(self, file_path: str) -> Dict[str, Any]:
        """Extract text, images, and tables from PDF."""
        content = {
            'text': '',
            'tables': [],
            'images': [],
            'metadata': {}
        }
        
        try:
            # Extract text
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                # Convert metadata to simple types for ChromaDB compatibility
                content['metadata'] = {
                    'pages': str(len(pdf_reader.pages)),
                    'title': str(pdf_reader.metadata.get('/Title', '')),
                    'author': str(pdf_reader.metadata.get('/Author', '')),
                    'subject': str(pdf_reader.metadata.get('/Subject', ''))
                }
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    content['text'] += f"\n--- Page {page_num + 1} ---\n{text}"
            
            # Extract tables using tabula
            try:
                tables = tabula.read_pdf(file_path, pages='all')
                for i, table in enumerate(tables):
                    if not table.empty:
                        content['tables'].append({
                            'table_index': i,
                            'data': table.to_dict('records'),
                            'columns': table.columns.tolist()
                        })
            except Exception as e:
                print(f"Warning: Could not extract tables from PDF: {e}")
                
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            
        return content
    
    def generate_image_description(self, image_path: str) -> str:
        """Generate description for an image using Gemini Vision."""
        try:
            image = Image.open(image_path)
            prompt = f"Describe this image in detail. Focus on the main content, objects, people, text, and any relevant information. Keep the description under {Config.MAX_DESCRIPTION_LENGTH} characters."
            
            response = self.vision_model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            print(f"Error generating image description: {e}")
            return "Image description could not be generated."
    
    def generate_table_summary(self, table_data: List[Dict], columns: List[str]) -> str:
        """Generate summary for table data using Gemini."""
        try:
            table_str = f"Columns: {', '.join(columns)}\n"
            table_str += "Data:\n"
            for i, row in enumerate(table_data[:10]):  # Limit to first 10 rows for summary
                table_str += f"Row {i+1}: {row}\n"
            
            if len(table_data) > 10:
                table_str += f"... and {len(table_data) - 10} more rows"
            
            prompt = f"Summarize this table data in a concise way. Focus on the main patterns, key information, and what this data represents. Keep it under {Config.MAX_SUMMARY_LENGTH} characters.\n\n{table_str}"
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating table summary: {e}")
            return "Table summary could not be generated."
    
    def chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None, 
                   separators: List[str] = None) -> List[str]:
        """Chunk text using recursive character text splitter."""
        chunk_size = chunk_size or Config.DEFAULT_CHUNK_SIZE
        chunk_overlap = chunk_overlap or Config.DEFAULT_CHUNK_OVERLAP
        separators = separators or Config.DEFAULT_SEPARATORS
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )
        
        return text_splitter.split_text(text)
    
    def process_csv_text_only(self, file_path: str) -> List[Dict[str, Any]]:
        """Process CSV file with text columns only, treating each row as a chunk."""
        try:
            df = pd.read_csv(file_path)
            chunks = []
            
            for index, row in df.iterrows():
                # Combine all text columns
                text_content = " ".join([str(val) for val in row.values if pd.notna(val)])
                
                chunk = {
                    'content': text_content,
                    'metadata': {
                        'row_index': index,
                        'columns': df.columns.tolist(),
                        'row_data': row.to_dict()
                    }
                }
                chunks.append(chunk)
            
            return chunks
        except Exception as e:
            print(f"Error processing CSV file: {e}")
            return []
    
    def get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text using Gemini."""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [] 