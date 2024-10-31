import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI, AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":
    embeddings = AzureOpenAIEmbeddings(
        model = os.getenv('MODEL_NAME'),
        azure_endpoint = os.getenv('API_BASE'),
        api_key= os.getenv('API_KEY'),
        openai_api_version= os.getenv('API_VERSION')
    )
    

    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv('api_base'),
        openai_api_version=os.getenv('api_version'),
        deployment_name=os.getenv('deployment_name'),
        openai_api_key=os.getenv('api_key'),
        openai_api_type="azure",
    )

    ## Save Document
    # Ingesting
    pdf_path = "./ReAct_paper.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    
    # Splitting
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)
    print(docs)

    # embeddings & save Vectorstore
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")



    ## load Document & QA
    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(), retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(res["answer"])