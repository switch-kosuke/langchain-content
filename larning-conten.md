# learning about Langchain

## Setup
- 仮想環境の作成
    > python -m venv langchain-venv  
      source langchain-venv/bin/activate  
      pip install langchain  
      pip install langchain-openai ->OpenAIが提供するサードパーティー  
      pip install langchain-community ->コミュニティで開発したコードが含まれる。
      pip install langchain-hub ->コミュニティから提供されたコードを動的に扱うことができる。

## About Langchain  
- Langchainとは
    Langchainは、言語モデルを使用するためのフレームワークである。  

- Chainとは  
    ユーザーからの問い合わせをアクションとして、PDFやネット検索を行う。  
    このアクションの連鎖をChainという

    - なぜLangchainのライブラリがこんなに多いのか？  
    一般的には、Google SDKやOpenaAI SDK等の数多くのソフトウェア開発キットが必要になる。  
    それらを包括的に扱うのが、このLnagchainである。