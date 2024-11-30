import os
import argparse
import shutil
import uuid
import yaml
from dotenv import load_dotenv
from langchain.schema.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores.chroma import Chroma
# from langchain_chroma import Chroma

from util import get_embedding_function
from test_utils import load_pdf, sem_split_documents2


threshold = "interquartile"   #############설정 ("percentile", "standard_deviation", "interquartile", "gradient")

load_dotenv()
upstage_api_key = os.environ['UPSTAGE_API_KEY']
user_agent = os.environ['USER_AGENT']


# Get config
config_path = "./configs.yaml"
with open(config_path) as f:
    config = yaml.load(f, Loader = yaml.FullLoader)

chroma_path = config[f"CHROMA_PATH({threshold})"]   ############# 설정
data_path = config["DATA_PATH"]


def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()
        
    for file_name in os.listdir(data_path):
        file_path = data_path+file_name
        # Create (or update) the data store.
        documents = load_pdf(file_path)
        chunks = sem_split_documents2(documents,threshold)
        add_to_chroma(chunks)
        print(f"문서 {file_name}을 데이터베이스 chroma({threshold})에 추가")


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=get_embedding_function()
    )

    # Calculate IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "ewha.pdf:6:2"
    # Page Source : Page Number : Chunk Index : Chunk Size : Unique ID

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        chunk_size = chunk.metadata.get("chunk_size")
    
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        unique_id = uuid.uuid4().hex[:8] # in case there's a chunk that has the same size in the same page 
        chunk_id = f"{current_page_id}:{current_chunk_index}:{chunk_size}:{unique_id}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)


if __name__ == "__main__":
    main()
