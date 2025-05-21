from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from typing import Any, Dict, List
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
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
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
    model='gpt-4o',
)  

### GCP VertexAI
vertexai.init(project=os.getenv('PROJECT_ID'), location=os.getenv('LOCATION'))
embeddings = VertexAIEmbeddings(model_name="text-embedding-005")

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    docsearch = PineconeVectorStore(index_name=os.getenv('INDEX_NAME'), embedding=embeddings)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )
    
    print(qa)
    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result


if __name__ == "__main__":
    text = "LangChainって何？"
    res = run_llm(query=text)
    print(res["answer"])

