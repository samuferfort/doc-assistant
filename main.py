from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import traceback
import logging
from dotenv import load_dotenv
from typing import Optional
import json
from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Document Processor")

# Add CORS middleware for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
)

@app.get("/health")
async def health_check():
    """Check if the service is running"""
    return {"status": "healthy"}

def extract_text_with_pypdf(file_path):
    """Extract text using PyPDF as a fallback method"""
    logger.info("Attempting text extraction with PyPDF")
    try:
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        return text
    except Exception as e:
        logger.error(f"PyPDF extraction error: {str(e)}")
        return ""

@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    strategy: str = Form(default="fast", regex="^(fast|hi_res)$"),
):
    """Process a PDF into chunks and embeddings for Next.js to store in Pinecone"""
    # Parse metadata if provided
    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            metadata_dict = {}

    # Add project_id and filename to metadata
    if project_id:
        metadata_dict["project_id"] = project_id
    metadata_dict["filename"] = file.filename

    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename)[1]
        ) as temp:
            content = await file.read()
            temp.write(content)
            temp_path = temp.name

        try:
            logger.info(f"Processing file: {file.filename} with strategy: {strategy}")
            logger.info(f"Temp file path: {temp_path}")
            
            documents = []
            
            # First try with UnstructuredLoader
            try:
                loader = UnstructuredLoader(
                    temp_path,
                    strategy=strategy,
                    chunking_strategy="by_title",
                    max_characters=2000,
                    new_after_n_chars=1500,
                )
                logger.info("UnstructuredLoader initialized successfully")
                
                documents = loader.load()
                logger.info(f"Document loading complete. Extracted {len(documents)} chunks.")
                
            except Exception as loader_error:
                logger.error(f"Error during document loading: {str(loader_error)}")
                logger.error(traceback.format_exc())
            
            # If UnstructuredLoader failed or returned no chunks, try PyPDF
            if len(documents) == 0:
                logger.info("UnstructuredLoader returned no chunks, trying PyPDF fallback")
                extracted_text = extract_text_with_pypdf(temp_path)
                
                if extracted_text:
                    logger.info(f"PyPDF extracted {len(extracted_text)} characters")
                    
                    # Split the text into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=2000,
                        chunk_overlap=200,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    
                    # Create document chunks
                    from langchain_core.documents import Document
                    full_doc = Document(page_content=extracted_text, metadata=metadata_dict)
                    documents = text_splitter.split_documents([full_doc])
                    
                    logger.info(f"Split into {len(documents)} chunks using text splitter")
                else:
                    logger.warning("PyPDF fallback also failed to extract text")

            # Prepare chunks and batch embeddings
            chunks = []
            texts = []
            for i, doc in enumerate(documents):
                doc.metadata.update(metadata_dict)
                # Add a unique chunk ID for Pinecone
                chunk_id = f"{project_id}_{file.filename}_{i}"
                chunks.append(
                    {"id": chunk_id, "text": doc.page_content, "metadata": doc.metadata}
                )
                texts.append(doc.page_content)

            # Check if we have any text to process
            if not texts:
                logger.warning("No text could be extracted from the document")
                return {
                    "chunks": [],
                    "embeddings": [],
                    "count": 0,
                    "warning": "No text could be extracted from the document"
                }

            logger.info(f"Generating embeddings for {len(texts)} chunks")
            # Batch embed all texts
            embeddings_list = embeddings.embed_documents(texts)
            logger.info("Embedding generation complete")

            # Return structured response
            return {
                "chunks": chunks,
                "embeddings": embeddings_list,
                "count": len(documents),
            }
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"Temporary file {temp_path} removed")

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)