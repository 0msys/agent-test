import chainlit as cl
import time
from typing import Any, Dict, List

from chainlit.context import context_var
from literalai import ChatGeneration, CompletionGeneration
from langchain.callbacks.tracers.schemas import Run
from datetime import datetime
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.agents import AgentAction, AgentFinish

# _on_run_updateメソッドをオーバーライドし、メッセージの表示を適正化する
class CustomAgentCallbackHandler(cl.LangchainCallbackHandler):
    def _on_run_update(self, run: Run) -> None:
        """Process a run upon update."""
        context_var.set(self.context)

        ignore, parent_id = self._should_ignore_run(run)

        if ignore:
            return

        current_step = self.steps.get(str(run.id), None)

        if run.run_type == "llm" and current_step:
            provider, model, tools, llm_settings = self._build_llm_settings(
                (run.serialized or {}), (run.extra or {}).get("invocation_params")
            )
            generations = (run.outputs or {}).get("generations", [])
            generation = generations[0][0]
            variables = self.generation_inputs.get(str(run.parent_run_id), {})
            text = generation.get("text")
            message = generation.get("message")
            tool_calls = message["kwargs"]["additional_kwargs"].get(
                    "tool_calls", []
                )
            if tool_calls: # tool_callsがある場合
                chat_start = self.chat_generations[str(run.id)]
                duration = time.time() - chat_start["start"]
                if duration and chat_start["token_count"]:
                    throughput = chat_start["token_count"] / duration
                else:
                    throughput = None

                message_completion = tool_calls[0] # なぜかDictを入れないといけないので、tool_calls[0]を入れておく

                current_step.generation = ChatGeneration(
                    provider=provider,
                    model=model,
                    tools=tools,
                    variables=variables,
                    settings=llm_settings,
                    duration=duration,
                    token_throughput_in_s=throughput,
                    tt_first_token=chat_start.get("tt_first_token"),
                    messages=[
                        self._convert_message(m) for m in chat_start["input_messages"]
                    ],
                    message_completion=message_completion,
                )
                tool_calls_message = text + "\n\n"
                for tool_call in tool_calls:
                    tool_calls_message += f"- [{tool_call["function"]["name"]}] ({tool_call["function"]["arguments"]})\n"
                current_step.name = "Tool Calls" # Stepの名前をわかりやすく変更
                current_step.output = tool_calls_message
            else:
                completion_start = self.completion_generations[str(run.id)]
                completion = generation.get("text", "")
                duration = time.time() - completion_start["start"]
                if duration and completion_start["token_count"]:
                    throughput = completion_start["token_count"] / duration
                else:
                    throughput = None
                current_step.generation = CompletionGeneration(
                    provider=provider,
                    model=model,
                    settings=llm_settings,
                    variables=variables,
                    duration=duration,
                    token_throughput_in_s=throughput,
                    tt_first_token=completion_start.get("tt_first_token"),
                    prompt=completion_start["prompt"],
                    completion=completion,
                )
                current_step.output = completion

            if current_step:
                current_step.end = datetime.utcnow().isoformat()
                self._run_sync(current_step.update())

            if self.final_stream and self.has_streamed_final_answer:
                if self.final_stream.content:
                    self.final_stream.content = completion
                self._run_sync(self.final_stream.update())

            return

        outputs = run.outputs or {}
        output_keys = list(outputs.keys())
        output = outputs
        if output_keys:
            output = outputs.get(output_keys[0], outputs)

        if current_step:
            current_step.input = run.serialized
            current_step.output = output
            current_step.end = datetime.utcnow().isoformat()
            self._run_sync(current_step.update())


class StreamingCallbackHandler(BaseCallbackHandler):
    async def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Run when Chat Model starts running."""
        step = cl.user_session.get("agent_thought_step")
        if step:
            step.output = step.output + "┃"
            await step.update()
        else:
            # Streaming用のStepがない場合は新しく作成
            async with cl.Step(name="Agent Thought", type="llm", root=True) as step:
                step.output = ""
            cl.user_session.set("agent_thought_step", step)

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if token:
            step = cl.user_session.get("agent_thought_step")
            if step:
                streaming_text = step.output[:-1] + token # "┃"を削除して、新しいtokenを追加
                step.output = streaming_text + "┃"
                await step.update()

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        step = cl.user_session.get("agent_thought_step")
        if step:
            step.output = step.output[:-1] + "\n" # "┃"を削除して、改行を追加
            await step.update()
    
    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        step = cl.user_session.get("agent_thought_step")
        if step:
            # Tool Callの情報を表示
            # tool_inputが長い場合は、省略して表示
            MAX_PREVIEW_LENGTH = 50
            tool_input_preview = action.tool_input if len(str(action.tool_input)) <= MAX_PREVIEW_LENGTH else str(action.tool_input)[:MAX_PREVIEW_LENGTH] + "..."
            step.output = step.output + f"- **🛠️ Tool Call:** {action.tool}({tool_input_preview})\n\n"
            await step.update()
    
    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        step = cl.user_session.get("agent_thought_step")
        if step:
            # Agentの終了時に、Streaming用のStepを削除
            time.sleep(1) # すぐに削除せず、少し待つ。好みで調整
            await step.remove()
            cl.user_session.set("agent_thought_step", None)

