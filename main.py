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

# Add CORS middleware to allow requests from your Next.js app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set to your Vercel app URL
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
    """Simple endpoint to check if the service is running"""
    return {"status": "healthy"}


@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
):
    """Process a document and return chunks with embeddings"""
    # Parse metadata if provided
    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            metadata_dict = {}

    # Add project_id to metadata
    if project_id:
        metadata_dict["project_id"] = project_id

    # Add filename to metadata
    metadata_dict["filename"] = file.filename

    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename)[1]
        ) as temp:
            # Write uploaded file content to temp file
            content = await file.read()
            temp.write(content)
            temp_path = temp.name

        try:
            # Process with Unstructured
            loader = UnstructuredLoader(
                temp_path,
                strategy="hi_res",
                chunking_strategy="by_title",
                max_characters=2000,
                new_after_n_chars=1500,
            )

            # Load and process document
            documents = loader.load()

            # Generate embeddings for each chunk
            document_chunks = []
            embeddings_list = []

            for doc in documents:
                # Add metadata to chunk
                doc.metadata.update(metadata_dict)

                # Generate embedding
                embedding = embeddings.embed_query(doc.page_content)

                # Add to results
                document_chunks.append(
                    {"text": doc.page_content, "metadata": doc.metadata}
                )
                embeddings_list.append(embedding)

            # Return processed chunks and embeddings
            return {
                "chunks": document_chunks,
                "embeddings": embeddings_list,
                "count": len(documents),
            }

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )


# Run the app if this file is executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
