import chainlit as cl

from langchain.agents import AgentExecutor, load_tools, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


@cl.on_chat_start
def start():
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
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    # Agentを作成
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, memory=memory
    )

    # AgentExecutorをセッションに保存
    cl.user_session.set("agent_executor", agent_executor)


@cl.on_message
async def main(message: cl.Message):

    # AgentExecutorをセッションから取得
    agent_executor = cl.user_session.get("agent_executor")  # type: AgentExecutor

    # Agentを実行
    # configでLangchainCallbackHandlerを指定することで途中経過を表示
    res = await agent_executor.ainvoke(
        {"input": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    )

    # Agentの出力を表示
    await cl.Message(content=res["output"]).send()
