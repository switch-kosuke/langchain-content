from dotenv import load_dotenv
load_dotenv()
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_ollama import ChatOllama

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
import os
from output_parsers import summary_parser, Summary

from typing import Tuple


def ice_break_with(name: str) -> Tuple[Summary, str]:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username, mock=True)

    summary_template = """
    given the Linkedin information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    \n{format_instructions}
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], 
        template=summary_template, 
        partial_variables={"format_instructions":summary_parser.get_format_instructions}
    )

    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv('api_base'),
        openai_api_version=os.getenv('api_version'),
        deployment_name=os.getenv('deployment_name'),
        openai_api_key=os.getenv('api_key'),
        openai_api_type="azure",
    )

    # chain = summary_prompt_template | llm
    chain = summary_prompt_template | llm | summary_parser

    res:summary = chain.invoke(input={"information": linkedin_data})

    print(res)

    return res, linkedin_data.get("profile_pic_url")


if __name__ == "__main__":
    load_dotenv()

    print("Ice Breaker Enter")
    ice_break_with(name="Eden Marco")