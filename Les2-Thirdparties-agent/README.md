# Create Agent App with GenerativeAI

## About Agent App
本セッションでは、LinkedInの情報を要約して表示する。  

### スクレイピングした情報からLLMに回答してもらう（Linkedin.py）
- Linkedinをスクレイピングする  
    Linkedinの標準APIは、扱いが難しい。  
    そこで、今回は[Proxycurl](https://nubela.co/proxycurl/linkedin)を用いて、スクレイピングする。  

- スクレイピングしたjsonをきれいにする
    jsonをきれいにするために、[Jsonlint](https://jsonlint.com/)を用いた。（これは何でもいい）

    ちなみに、スクレイピングしたjsonは、[GitHub gist](https://gist.github.com/)で公開することもできる。

## Topic about LangChain & LLM
LLMは、世界中のあらゆるデータにアクセスできるスーパーパワーを備えている。  
LangChainは、これらの世界中のデータを繋ぐための強力なフレームワークである。  

LangChainに接続したLLMは、ユーザーの要望に対して最適な外部APIやデータベースの検索を実施する(Agent)。  
これらのAgentは、思考の連鎖とREACTによって実現している。（詳細は、後）  
    + 思考の連鎖  
        接続した外部データに対して、ユーザーの要望に答える情報を検索する技術
    + REACT
        外部データへの検索完了すれば、それらの推論結果を返す技術

### create Agent (LLM can use tool with searching Linkedin URL)

### 参考
- 思考の連鎖：https://arxiv.org/pdf/2201.11903v1
- REACT：https://arxiv.org/pdf/2210.03629