# learning about Langchain

## Setup 📝
- 仮想環境の作成
    ```tarminal
    python -m venv langchain-venv  
    source langchain-venv/bin/activate  
    pip install langchain  
    pip install langchain-openai ->OpenAIが提供するサードパーティー  
    pip install langchain-community ->コミュニティで開発したコードが含まれる。
    pip install langchain-hub ->コミュニティから提供されたコードを動的に扱うことができる。
    ```

- 環境変数の処理（APIキー等）  
    + .envファイルの作成  
    この中にAPIキー等を記載  
    
    + python-dotenvライブラリを取得  
    以下のように記載  
    ![こんな感じ](/langchain-content/images/dotenv_ex.png)

## About Langchain ⚙  
- Langchainとは  
    Langchainは、言語モデルを使用するためのフレームワークである。  

- Chainとは  
    ユーザーからの問い合わせをアクションとして、PDFやネット検索を行う。  
    このアクションの連鎖をChainという

    - なぜLangchainのライブラリがこんなに多いのか？  
    一般的には、Google SDKやOpenaAI SDK等の数多くのソフトウェア開発キットが必要になる。  
    それらを包括的に扱うのが、このLnagchainである。

## 簡単なLangChain Prompt  
- PromptTemplate  
    プロンプトテンプレートを用いることで、単なるテキスト内に書き換え可能な変数を入れることができる  
    ```python
    summary_template = """
        こんにちは, {information}
        """

    summary_prompt_template = PromptTemplate(input_variables="information", template=summary_template)
    ```

