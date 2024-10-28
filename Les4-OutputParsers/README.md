# Output Parsers

## 1. 概要

LLMからの出力は、通常Text形式であるが、これらの出力がシステムにとって不便なことも多々ある.  
そこで、LangChainオブジェクトの一つである「Output Parsers」では、これらの出力を適した形式のテーブル(json, xml, csv etc,.)に変換してくれる.  

参考  
- https://python.langchain.com/docs/concepts/output_parsers/
- https://python.langchain.com/docs/how_to/output_parser_structured/

## 2. Code


## 2.2. output_parsers.py
```python
from typing import List, Dict, Any

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
```

- Pydantic  
    Pythonのデータクラスト類似する外部ライブラリ.  
    データ検証や設定管理によく用いられる.今回はLLMからの出力にはこの「Pydantic」を用いる.  

```python
class Summary(BaseModel):
    summary: str = Field(description="summary")
    facts: List[str] = Field(description="interesting facts about them")

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary, "facts": self.facts}

summary_parser = PydanticOutputParser(pydantic_object=Summary)
```

## 2.3. ice_breaker.py  

```python
summary_template = """
    given the Linkedin information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    \n{format_instructions}
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], 
        template=summary_template, 
        partial_variables={"format_instructions":summary_parser.get_format_instructions()}
    )
```

プロンプトテンプレートに対して、OutputParserのスキーマデータを部分変数で提供する.  

- get_format_instructions  
    LangChainのPydanticオブジェクトを受け取る関数である.  
    この受け取ったオブジェクトをプロンプト内に貼り付ける.  

```python
chain = summary_prompt_template | llm | summary_parser
```

- LangChainの特別表現記法 LCL
    このパイプ演算子は、要約プロンプトのテンプレートをLLMに送り込み、LLMの出力を要約パーサーに送り込んでいる処理である.  
    後に解説あり.  

```bash
To find the LinkedIn profile page for Eden Marco, I will use the tool to crawl Google for the LinkedIn profile page.

Action: Crawl Google 4 linkedin profile page
Action Input: Eden Marco[{'url': 'https://www.udemy.com/user/eden-marco/', 'content': 'Eden Marco | LLM Specialist is a Udemy instructor with educational courses available for enrollment. Check out the latest courses taught by Eden Marco | LLM Specialist'}, {'url': 'https://www.udemy.com/course/langchain/', 'content': 'Eden Marco | LLM Specialist. Best Selling Instructor. 4.6 Instructor Rating. 21,403 Reviews. 90,628 Students. 7 Courses. I am a passionate Software Engineer with years of experience in back-end development, one of the first engineers at Orca Security, and now I am working as a Customer Engineer at Google Cloud.'}, {'url': 'https://www.linkedin.com/today/author/eden-marco', 'content': 'Check out professional insights posted by Eden Marco, LLMs @ Google Cloud | Best-selling Udemy Instructor | Backend &amp; GenAI | Opinions stated here are my own, not those of my company'}, {'url': 'https://www.linkedin.com/posts/eden-marco_one-of-my-favorite-things-about-being-a-udemy-activity-7176269229858381826-7U6d', 'content': 'Eden Marco LLMs @ Google Cloud | Best-selling Udemy Instructor | Backend & GenAI 1w Report this post Contextual Answers from the AI21 Labs team is a fantastic model when you want the LLM to always'}, {'url': 'https://github.com/emarco177/', 'content': 'in/eden-marco @EdenEmarco177; Achievements. x2. Achievements. x2. Block or Report. Block or report emarco177 Block user. Prevent this user from interacting with your repositories and sending you notifications. Learn more about blocking users. You must be logged in to block users.'}]Final Answer: https://www.linkedin.com/today/author/eden-marco

> Finished chain.
summary='Eden Marco is a Customer Engineer at Google, based in Tel Aviv, Israel. A best-selling instructor on Udemy, Eden has a strong background in backend development and has held various positions in the tech industry, including roles at Orca Security, Wizer, and Deep Instinct.' facts=['Eden Marco has produced and published two best-selling courses on Udemy, attracting over 9,000 students and receiving more than 800 ratings with a solid 4.7-star rating.', 'Eden served as a Captain in the Israel Defense Forces from July 2010 to August 2014.']

```
