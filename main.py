from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
from dotenv import load_dotenv
from typing import Optional
import json
from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings

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
            # Process with UnstructuredLoader
            loader = UnstructuredLoader(
                temp_path,
                strategy=strategy,
                chunking_strategy="by_title",
                max_characters=2000,
                new_after_n_chars=1500,
            )
            documents = loader.load()

            # Prepare chunks and batch embeddings
            chunks = []
            texts = []
            for i, doc in enumerate(documents):
                doc.metadata.update(metadata_dict)
                # Add a unique chunk ID for Pinecone
                chunk_id = f"{project_id}_{file.filename}_{i}"
                chunks.append({
                    "id": chunk_id,
                    "text": doc.page_content,
                    "metadata": doc.metadata
                })
                texts.append(doc.page_content)

            # Batch embed all texts
            embeddings_list = embeddings.embed_documents(texts)

            # Return structured response
            return {
                "chunks": chunks,
                "embeddings": embeddings_list,
                "count": len(documents)
            }

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)