# Building a Document Assistant Apps
このセクションでは、ドキュメントアシストのアプリケーションを０から作成していく.  

作成目標：https://github.com/emarco177/documentation-helper/tree/main

##  概要
以下の内容を扱う.  
1. Vector Databases (Ingestion)
2. RetrieveQA chain (RAG)
3. Similarity Search
4. LLM Memory
5. Streamlit

## Lectures
### 0. Environment Setup  
1. Download about LangChain Document
    今回Documentの対象は、[このサイト](https://python.langchain.com/v0.1/docs/expression_language/get_started/).  
    [Download Data](./langchain-docs/)

    これらのサイトをダウンロードしただけのデータには、ゴミ（HTMLタグやその他）がたくさん含まれている. しかし、これらのゴミをLLMに送信する際に処理する必要はない.  
    なぜならこれらのごみを含む情報を全て、LLMが処理してくれるからだ.  

2. Pinecorn  
    [こちら](https://www.pinecone.io/)でログイン⇒新規Indexを作成⇒envファイルにAPIキーを設置  

3. 必要なライブラリをインストール  
    ```bash
    python -m venv venv
    source venv/bin/activate
    pipenv install
    ```
    参考：https://qiita.com/tamazoo/items/9fc0dfda3c583583c402  

### 1. Vector Databases (Ingestion) -> ingestion.py
1. ライブラリ  
    ```python
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import ReadTheDocsLoader
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    ```

    - [ReadTheDocsLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.readthedocs.ReadTheDocsLoader.html)  
        Read the Docsは、GitHubリポジトリのドキュメント作成を支援するツール. langchainでは、このドキュメントをロードする仕組みをReadtheDocsLoaderとして提供している.  

2. Contents
    ```python
    def ingest_docs():
        ### Docs Loading
        loader = ReadTheDocsLoader("langchain-docs/langchain.readthedocs.io/en/v0.1")
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
        PineconeVectorStore.from_documents(
            documents, embeddings, index_name="langchain-doc-index"
        )
        print("****Loading to vectorstore done ***")
    ```

    - RecursiveCharacterTextSplitter  
        テキスト分割を行う関数. LLMには、ユーザーの質問と、このテキストチャンクを追加したコンテキストを送信する.  
        この時、短く簡潔な答えが必要な場合には、チャンクサイズは小さい方が良い. しかし、チャンクサイズが極端に小さくなると意味を持たなくなるので、避けた方が良い.  

    - チャンク(ドキュメント)ごとに新規URLを設定  
        これによってこのドキュメントが何処から来たのか、完全なURLを教えてくれる.  

### 2. 2. RetrieveQA chain (RAG) ->backend/core.py
質問に関連する文章を取得し、検索された文章で質問を補強し回答を生成する.  

1. ライブラリ  
    ```python
    from dotenv import load_dotenv
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain import hub
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_pinecone import PineconeVectorStore
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    ```

    - create_retrieval_chain  
        関連する文章を取得する機能を実装したオブジェクト. このチェーンは質問文と関連するドキュメントを同時にLLMに送信する事が出来る.  

    - create_stuff_documents_chain  
        コンテキストをプロンプトに差し込む処理をする.  

2. contents
    ```python

# Trouble shootings
1. エンベディングモデルのレート制限を超過
    - エラー内容  
        ```python
        openai.RateLimitError: Error code: 429 - {'error': {'code': '429', 'message': 'Requests to the Embeddings_Create Operation under Azure OpenAI API version 2024-08-01-preview have exceeded call rate limit of your current OpenAI S0 pricing tier. Please retry after 86400 seconds. Please go here: https://aka.ms/oai/quotaincrease if you would like to further increase the default rate limit.'}}
        ```

    - 解決方法  
        ①APIのレートティアを上げる.  
        ②分割してベクトル化.  
        ③別のエンベディングモデルを使用（今回はこちらを採用）

2. Pineconeのディメンションが異なる.  
    - エラー内容  
        ```bash
        pinecone.core.client.exceptions.PineconeApiException: (400)
        Reason: Bad Request
        HTTP response headers: HTTPHeaderDict({'Date': 'Fri, 01 Nov 2024 03:15:09 GMT', 'Content-Type': 'application/json', 'Content-Length': '103', 'Connection': 'keep-alive', 'x-pinecone-request-latency-ms': '902', 'x-pinecone-request-id': '9069753074993681914', 'x-envoy-upstream-service-time': '48', 'server': 'envoy'})
        HTTP response body: {"code":3,"message":"Vector dimension 768 does not match the dimension of the index 1536","details":[]}
        ```

    - 解決方法  
        Pineconeの作成したIndexのディメンションサイズを、エンベディングモデルに合わせて瀬ってする必要がある.  
        参考：[LINK](https://medium.com/@nunocarvalhodossantos/i-tried-pinecone-and-this-was-what-happened-a-guide-through-console-and-python-code-cf668b3d273b)