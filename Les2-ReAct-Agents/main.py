from typing import Union, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool, tool
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser

import os

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non alphabetic characters just in case

    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")


if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought:
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    # llm = ChatOpenAI(temperature=0, stop = ["\nObservation", "Observation"])

    llm = AzureChatOpenAI(
        temperature=0, stop = ["\nObservation", "Observation"],
        azure_endpoint=os.getenv('api_base'),
        openai_api_version=os.getenv('api_version'),
        deployment_name=os.getenv('deployment_name'),
        openai_api_key=os.getenv('api_key'),
        openai_api_type="azure",
    )

    intermediate_steps = []
    agent = (
        {
            "input": lambda x: x["input"],
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    # res = agent.invoke({"input":"What is the length of 'DOG' in characters"})
    # print(res)

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of 'DOG' in characters?",
        }
    )
    print(agent_step)

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"{observation=}")

