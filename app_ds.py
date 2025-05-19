from typing import List, cast

import chainlit as cl
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage, ToolCallExecutionEvent
from autogen_core import CancellationToken
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool

from autogen_agentchat.conditions import TextMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.base import TaskResult

import os
from dotenv import load_dotenv

# Load from .env file
load_dotenv()

@cl.set_starters  # type: ignore
async def set_starts() -> List[cl.Starter]:
    return [
        cl.Starter(
            label="Greetings",
            message="Hello! What can you help me with today?",
        ),
    ]


@cl.on_chat_start  # type: ignore
async def start_chat() -> None:
    # Load model configuration and create the model client.

    api_key = os.getenv("ANTHROPIC_KEY")
    model_id = os.getenv("MODEL_NAME")
    model_client = AnthropicChatCompletionClient(model=model_id, api_key=api_key)

    code_tool = PythonCodeExecutionTool(LocalCommandLineCodeExecutor(work_dir="coding"))

    # Create the assistant agent with the get_weather tool.
    assistant = AssistantAgent(
        name="assistant",
        tools=[code_tool],
        model_client=model_client,
        system_message="""You are a helpful assistant. At one go you perform only one of below task -
        1. Perform Exploratory Data anayalsis : When user gives a dataset use coding tool to read and prepare basic stats and plots. Save them and share the path with user. Once completed then close the conversation by saying "TERMINATE".
        2. Train classification model : When user gives a dataset use coding tool to read and train a xgboost model. Save the model and train-test results with plots and share the path with user. Once completed then close the conversation by saying "TERMINATE".

        Note : Whenever the task is complete you must reply with "TERMINATE" . 
        """,
        model_client_stream=True,  # Enable model client streaming.
        reflect_on_tool_use=False,  # Reflect on tool use.
    )

    termination_condition = TextMentionTermination("TERMINATE")

    # Create a team with the looped assistant agent and the termination condition.
    team = RoundRobinGroupChat(
        [assistant],
        termination_condition=termination_condition,
    )

    # Set the assistant agent in the user session.
    cl.user_session.set("prompt_history", "")  # type: ignore
    cl.user_session.set("agent", team)  # type: ignore


@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    # Get the team from the user session.
    team = cast(RoundRobinGroupChat, cl.user_session.get("agent"))  # type: ignore
    # Streaming response message.
    streaming_response: cl.Message | None = None
    # Stream the messages from the team.
    async for msg in team.run_stream(
        task=[TextMessage(content=message.content, source="user")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(msg, ModelClientStreamingChunkEvent):
            # Stream the model client response to the user.
            if streaming_response is None:
                # Start a new streaming response.
                streaming_response = cl.Message(content=msg.source + ": ", author=msg.source)
            await streaming_response.stream_token(msg.content)
        elif streaming_response is not None:
            # Done streaming the model client response.
            # We can skip the current message as it is just the complete message
            # of the streaming response.
            await streaming_response.send()
            # Reset the streaming response so we won't enter this block again
            # until the next streaming response is complete.
            streaming_response = None
        elif isinstance(msg, TaskResult):
            # Send the task termination message.
            final_message = "Task terminated. "
            if msg.stop_reason:
                final_message += msg.stop_reason
            await cl.Message(content=final_message).send()
        else:
            # Skip all other message types.
            pass