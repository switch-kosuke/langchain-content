from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import vertexai
from langchain_google_vertexai import VertexAIEmbeddings
import os

load_dotenv()

## Azure OpenAI
# embeddings = AzureOpenAIEmbeddings(
#     model = os.getenv('MODEL_NAME'),
#     azure_endpoint = os.getenv('API_BASE'),
#     api_key= os.getenv('API_KEY'),
#     openai_api_version= os.getenv('API_VERSION')
# )

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('api_base'),
    openai_api_version=os.getenv('api_version'),
    deployment_name=os.getenv('deployment_name'),
    openai_api_key=os.getenv('api_key'),
    openai_api_type="azure",
)

### GCP VertexAI
vertexai.init(project=os.getenv('PROJECT_ID'), location=os.getenv('LOCATION'))
embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

def run_llm(query: str):
    docsearch = PineconeVectorStore(index_name=os.getenv('INDEX_NAME'), embedding=embeddings)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain
    )
    print(qa)
    result = qa.invoke(input={"input": query})
    return result


if __name__ == "__main__":
    text = "LangChainって何？"
    res = run_llm(query=text.encode('utf-8'))
    print(res["answer"])

