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
import pdfplumber
import camelot
import tenacity

from config import Config
from PyPDF2 import PdfReader

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

    def extract_tables(self, path):
        """Input - pdf path. Output - List of tables (data frame)"""
        tables_cam = camelot.read_pdf(path, pages="all")
        camelot_tables = []
        for table in tables_cam:
            table_df = table.df
            table_df = table_df.dropna(how='all').replace(r'^\s*$', None, regex=True).dropna(how='all')
            table_df.dropna(axis=1, how="all", inplace=True)
            if table_df.shape[0]>1 and table_df.shape[1]>1:
                page_num = table.page
                page_tag = pd.DataFrame([[f"page: {page_num}"]*table_df.shape[1]], columns=table_df.columns)
                table_df_fin = pd.concat([page_tag, table_df], ignore_index=True)
                camelot_tables.append(table_df_fin)
        return camelot_tables

    def extract_pdf_content(self, file_path: str, image_output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Extract text, tables, and images from PDF using pdfplumber and camelot for tables."""
        content = {
            'text': '',
            'tables': [],
            'images': [],
            'metadata': {}
        }
        try:
            print(f"[PDF PROCESS] Extracting text from: {file_path}")
            with pdfplumber.open(file_path) as pdf:
                # Metadata
                meta = pdf.metadata or {}
                content['metadata'] = {
                    'pages': str(len(pdf.pages)),
                    'title': str(meta.get('Title', '')),
                    'author': str(meta.get('Author', '')),
                    'subject': str(meta.get('Subject', ''))
                }
                # Extract text
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ''
                    content['text'] += f"\n--- Page {page_num + 1} ---\n{text}"
            print(f"[PDF PROCESS] Extracting tables from: {file_path}")

            # Extract tables using Camelot
            camelot_tables = self.extract_tables(file_path)
            for idx, table_df in enumerate(camelot_tables):
                columns = table_df.columns.tolist()
                data = table_df.to_dict(orient='records')
                content['tables'].append({
                    'table_index': str(idx),
                    'data': data,
                    'columns': columns
                })
            print(f"[PDF PROCESS] Extracting images from: {file_path}")

            # Extract images using pdfplumber
            if image_output_dir is None:
                # Default to 'uploaded_files/pdf_images' in working directory
                image_output_dir = os.path.join(os.path.dirname(__file__), '..', 'uploaded_files', 'pdf_images')
            
            os.makedirs(image_output_dir, exist_ok=True)
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    for img_idx, img in enumerate(page.images):
                        image = page.to_image(resolution=150)
                        image_path = os.path.join(image_output_dir, f"page_{page_num+1}_img_{img_idx+1}.png")
                        image.save(image_path, format="PNG")
                        content['images'].append({
                            'page': page_num + 1,
                            'name': f"img_{img_idx+1}",
                            'image_path': image_path
                        })
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
        return content

    def generate_image_description(self, image_path: str) -> str:
        """Generate description for an image using Gemini Vision, with retry on rate limit."""
        try:
            image = Image.open(image_path)
            prompt = f"Describe this image in detail. Focus on the main content, objects, people, text, architectures, table information, numeric awareness and any relevant information. Keep the description under {Config.MAX_DESCRIPTION_LENGTH} characters."
            
            @tenacity.retry(
                wait=tenacity.wait_exponential(multiplier=1, min=1, max=30),
                stop=tenacity.stop_after_attempt(5),
                retry=tenacity.retry_if_exception_type(Exception)
            )
            def call_gemini():
                return self.vision_model.generate_content([prompt, image])
            response = call_gemini()
            return response.text
        except Exception as e:
            print(f"Error generating image description: {e}")
            return "Image description could not be generated."
    
    def generate_table_summary(self, table_data: List[Dict], columns: List[str]) -> str:
        """Generate summary for table data using Gemini."""
        try:
            columns_str = [str(col) for col in columns]
            table_str = f"Columns: {', '.join(columns_str)}\n"
            table_str += "Data:\n"
            for i, row in enumerate(table_data[:10]):  # Limit to first 10 rows for summary
                table_str += f"Row {i+1}: {row}\n"
            
            if len(table_data) > 10:
                table_str += f"... and {len(table_data) - 10} more rows"
            
            prompt = f"Summarize this table data in a concise way. Focus on the main patterns, key information, Numeric awareness, and what this data represents. Keep it under {Config.MAX_SUMMARY_LENGTH} characters.\n\n{table_str}"
            
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