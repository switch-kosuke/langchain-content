import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def format_docs(docs):
    return "\n\n" .join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print(" Retrieving...")

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

    query = "what is Pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    # result = retrival_chain.invoke(input={"input": query})

    # print(result)

    # オリジナルプロンプトで検証してみる
    template = """Use the following piece of context to answer the question at the end. 
    If you dont know the answer, just say that you dont know, dont try to make up an answer.

    {context}

    question: {question}

    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    rag = rag_chain.invoke(query)

    print(rag)