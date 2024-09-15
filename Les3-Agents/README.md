# Topic about LangChain & LLM
LLMは、世界中のあらゆるデータにアクセスできるスーパーパワーを備えている。  
LangChainは、これらの世界中のデータを繋ぐための強力なフレームワークである。  

LangChainに接続したLLMは、ユーザーの要望に対して最適な外部APIやデータベースの検索を実施する(Agent)。  
これらのAgentは、思考の連鎖とREACTによって実現している。（詳細は、後）  

    + 思考の連鎖  
        接続した外部データに対して、ユーザーの要望に答える情報を検索する技術
    
    + REACT
        外部データへの検索完了すれば、それらの推論結果を返す技術

## create Agent (LLM can use tool with searching Linkedin URL)
```python
from langchain import hub
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
```
- hub  
    コミュニティで事前に作られたプロンプトをダウンロードする  
    
- create_react_agent(REACT)  
    入力：Agentのパワーとなる外部リソース(ツール、プロンプト、リアクトプロンプト)をLLMが受け取る。  
    出力：REACTアルゴリズムに合わせた形式で返す。  

- AgentExecutor  
    プロンプトや指示を受け取るオブジェクト


### 参考
- 思考の連鎖：https://arxiv.org/pdf/2201.11903v1
- REACT：https://arxiv.org/pdf/2210.03629