# Building a Document Assistant Apps
このセクションでは、ドキュメントアシストのアプリケーションを０から作成していく.  

作成目標：https://github.com/emarco177/documentation-helper/tree/main

##  概要
以下の内容を扱う.  
1. Vector Databases
2. RetrieveQA chain
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

        