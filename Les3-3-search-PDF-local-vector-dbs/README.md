# search PDF (local vector DBs)
このセクションでは、自身のPDFに対してチャットする仕組みを構築する.  
また、今までのクラウドベースのベクターストア（Pinecorn）ではなく、ローカルベクターストアを導入する.  

今回は[ReActの論文](./ReAct_paper.pdf)を参照する

## Code
### set up
```bash 
pip install pypdf langchain langchain-openai langchain-community langchainhub faiss-cpu
```

### content
```python
    # Ingesting
    pdf_path = "/Users/edenmarco/Desktop/tmp/react.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    
    # Splitting
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    # embeddings & save Vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    # load VectorStore
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
```

    - FAISS  
        [Faiss](https://ai.meta.com/tools/faiss/)は、Facebookによって公開されたツール.  
        PDFやテキストファイルのようなオブジェクトを回転させ、類似検索を実行するのに役立つパッケージである. このようにして、LLMにもっと多くの情報を与える事が出来る.  
        これの凄いところは、変換されたベクターデータをRAMに収まるように変換すること.  

    - allow_dangerous_deserialization=True  
        危険なデシリアライズを許可することで、セキュリティリスクが増加する可能性あり.  
