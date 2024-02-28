import streamlit as st

from langchain.agents import AgentExecutor, load_tools, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


# Agentの作成

# モデルを初期化
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True)

# ツールをロード
tools = load_tools(["ddg-search"])

# プロンプトを作成
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# チャット履歴のメモリを作成
chat_history = StreamlitChatMessageHistory(key="chat_messages")

# Agentを作成
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


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
