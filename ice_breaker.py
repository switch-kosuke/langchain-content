from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

information = """
    ウォルト・ディズニー（Walt Disney、1901年12月5日 -1966年12月15日 ）は、アメリカ合衆国・イリノイ州シカゴに生まれたアニメーション作家、アニメーター、プロデューサー、映画監督、脚本家、漫画家、声優、実業家、エンターテイナー。

    ウォルト・ディズニーのサイン
    世界的に有名なアニメーションキャラクター「ミッキーマウス」をはじめとするキャラクターの生みの親で、『ディズニーリゾート』の創立者である。兄のロイ・O・ディズニーと共同で設立したウォルト・ディズニー・カンパニーは数々の倒産、失敗を繰り返すも、350億ドル以上の収入を持つ国際的な大企業に発展した。

    本名はウォルター・イライアス・ディズニー（Walter Elias Disney）。一族はアイルランドからの移民であり、姓の「ディズニー」（Disney）は元々「d'Isigny」と綴られ、フランスのノルマンディー地方のカルヴァドス県のイジニー＝シュル＝メール（フランス語版）から11世紀にイギリスやアイルランドに渡来したノルマン人の末裔であることに由来し、後に英語風に直され「ディズニー」となった。「イライアス」は父名。
"""

if __name__=="__main__":
    load_dotenv
    print("Hello world")
    # print(os.environ['OPENAI_API_KEY'])

    summary_template = """
        こんにちは, {information}
    """

    summary_prompt_template = PromptTemplate(input_variables="information", template=summary_template)

    llm = ChatOpenAI(temperature=0, model_name="gpt-.5-turbo")

    chain = summary_prompt_template | llm

    res = chain.invoke(input={"information": information})

    print(res)