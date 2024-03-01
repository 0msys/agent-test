import streamlit as st

from langchain.agents import AgentExecutor, tool
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.chains import LLMMathChain

# モデルを初期化
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True)
llm_math_chain = LLMMathChain.from_llm(llm=llm)


@tool
def calculator(expression: str) -> str:
    """Calculates the result of a mathematical expression."""
    return llm_math_chain.invoke(expression)


@tool
def ddg_search(query: str) -> str:
    """Searches DuckDuckGo for a query and returns the results."""
    search = DuckDuckGoSearchResults()
    return search.invoke(query)


# Agentの作成
# ツールをロード
tools = [calculator, ddg_search]

# プロンプトを作成
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant. You are multilingual, so adapt to the language of your users.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# チャット履歴のメモリを作成
chat_history = StreamlitChatMessageHistory(key="chat_messages")

# ツールをバインド
llm_with_tools = llm.bind_tools(tools)

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


# チャット履歴を表示
for chat in chat_history.messages:
    st.chat_message(chat.type).write(chat.content)

# チャットの表示と入力
if prompt := st.chat_input():

    # ユーザーの入力を表示
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        # StreamlitCallbackHandlerを使ってAgentの途中経過を表示
        st_callback = StreamlitCallbackHandler(st.container())

        # Agentを実行
        response = agent_executor.invoke(
            {"input": prompt, "chat_history": chat_history.messages},
            {"callbacks": [st_callback]},
        )

        # Agentの出力を表示
        st.write(response["output"])

    # チャット履歴を更新
    chat_history.add_messages(
        [
            HumanMessage(content=prompt),
            AIMessage(content=response["output"]),
        ]
    )
