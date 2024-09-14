from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from Linkedin import scrape_linkedin_profile

if __name__=="__main__":
    print("Hello LangChain")
    
    #### APIキーの取得
    load_dotenv()
    # api_key = os.enviton['OPENAI_API_KEY']
    api_key = os.environ['GEMINI_API_KEY']

    #### プロンプトのテンプレート作成と処理
    summary_template = """
        情報を基に、人物像を要約してください。
        情報：{information}
    """

    summary_prompt_template = PromptTemplate(input_variables="information", template=summary_template)

    #### LLMの初期設定
    # llm = ChatOpenAI(temperature=0, model_name="gpt-.5-turbo")
    # llm = ChatGoogleGenerativeAI(api_key=api_key, temperature=0, model="gemini-1.5-flash",max_output_tokens=20)
    llm = ChatOllama(model="llama3")

    #### LLMとプロンプトの結び付け
    # res = llm.invoke("Hello")  # シンプルレスポンス
    chain = summary_prompt_template | llm | StrOutputParser()
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/eden-marco/",
    )
    res = chain.invoke(input={"information": linkedin_data})

    #### 応答
    print(res)