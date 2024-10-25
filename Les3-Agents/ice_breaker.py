from dotenv import load_dotenv
load_dotenv()
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_ollama import ChatOllama

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
import os


def ice_break_with(name: str) -> str:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username)

    summary_template = """
    given the Linkedin information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv('api_base'),
        openai_api_version=os.getenv('api_version'),
        deployment_name=os.getenv('deployment_name'),
        openai_api_key=os.getenv('api_key'),
        openai_api_type="azure",
    )

    chain = summary_prompt_template | llm

    res = chain.invoke(input={"information": linkedin_data})

    print(res)


if __name__ == "__main__":
    load_dotenv()

    print("Ice Breaker Enter")
    ice_break_with(name="Eden Marco")