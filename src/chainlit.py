import chainlit as cl

from langchain.agents import AgentExecutor, tool
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.chains import LLMMathChain

from callbacks import CustomAgentCallbackHandler, StreamingCallbackHandler

# モデルを初期化
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True)
llm_math_chain = LLMMathChain.from_llm(llm=llm)


@tool
async def calculator(expression: str) -> str:
    """Calculates the result of a mathematical expression."""
    return llm_math_chain.invoke(expression)


@tool
async def ddg_search(query: str) -> str:
    """Searches DuckDuckGo for a query and returns the results."""
    search = DuckDuckGoSearchResults()
    return search.invoke(query)


@cl.on_chat_start
def start():

    # ツールをロード
    tools = [calculator, ddg_search]

    # プロンプトを作成
    MEMORY_KEY = "chat_history"
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are very powerful assistant. You are multilingual, so adapt to the language of your users.",
            ),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # ツールをバインド
    llm_with_tools = llm.bind_tools(tools)

    # チャット履歴を初期化
    chat_history = []
    cl.user_session.set("chat_history", chat_history)

    # Agentを作成
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    # AgentExecutorをセッションに保存
    cl.user_session.set("agent_executor", agent_executor)


@cl.on_message
async def main(message: cl.Message):

    # セッションから取得
    agent_executor = cl.user_session.get("agent_executor")
    chat_history = cl.user_session.get("chat_history")

    res = await agent_executor.ainvoke(
        {"input": message.content, "chat_history": chat_history},
        config=RunnableConfig(
            callbacks=[CustomAgentCallbackHandler(), StreamingCallbackHandler()]
        ),
    )

    # チャット履歴を更新
    chat_history.extend(
        [
            HumanMessage(content=message.content),
            AIMessage(content=res["output"]),
        ]
    )

    # Agentの出力を表示
    await cl.Message(content=res["output"]).send()
