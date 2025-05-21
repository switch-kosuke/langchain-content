from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
import vertexai
from langchain_google_vertexai import VertexAIEmbeddings

### Azure OpenAI
# embeddings = AzureOpenAIEmbeddings(
#     model = os.getenv('EMB_MODEL_NAME'),
#     azure_endpoint = os.getenv('API_BASE'),
#     openai_api_version= os.getenv('EMB_API_VERSION')
# )
# llm = AzureChatOpenAI(
#     azure_endpoint=os.getenv('api_base'),
#     openai_api_version=os.getenv('api_version'),
#     deployment_name=os.getenv('deployment_name'),
#     openai_api_key=os.getenv('api_key'),
#     openai_api_type="azure",
# )

### GCP VertexAI
vertexai.init(project=os.getenv('PROJECT_ID'), location=os.getenv('LOCATION'))
embeddings = VertexAIEmbeddings(model_name=os.getenv('MODEL_ID'))


def ingest_docs():
    
    ### Docs Loading
    loader = ReadTheDocsLoader("./langchain-docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    ### chunking split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    
    ### Save to Pinecorn
    print(f"Going to add {len(documents)} to Pinecone")
    print(f"ターゲットPinecone: {os.environ['INDEX_NAME']}")
    # PineconeVectorStore.from_documents(
    #     documents, embeddings, index_name=os.environ['INDEX_NAME']
    # )
    
    # バッチサイズを設定
    batch_size = 500
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    # バッチ処理
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{total_batches} ({len(batch)} documents)")
        
        # バッチごとにPineconeに保存
        PineconeVectorStore.from_documents(
            batch, embeddings, index_name=os.environ['INDEX_NAME']
        )
        
        print(f"Batch {i//batch_size + 1}/{total_batches} completed")
    
    
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()