"""
fakes.py - pretend AI models for testing WITHOUT internet and WITHOUT API keys.

Why do we need this?
    A real AI model (Gemini, OpenAI, Claude, Ollama) gives a slightly
    different answer every time, needs a key, and costs quota.
    For tests we want the SAME answer every time, for free, offline.
    So we "script" the answers ourselves.

What is inside:
    LangChain fakes (work in every episode):
        fake_llm(...)          -> a chat model that replies with your scripted answers
        tool_call(...)         -> a scripted reply where the model "asks" to use a tool
        structured_reply(...)  -> a scripted reply for llm.with_structured_output(...)
        gemini_style_reply(...) -> a reply shaped like Gemini 3+ (content is a list of blocks)
        fake_embeddings()      -> offline embeddings for vector store / RAG tests

    CrewAI fakes (only when CrewAI is installed, episodes 13-14):
        FakeCrewAILLM(responses=[...])
        react_final_answer(...)  -> text that makes a CrewAI agent finish
        react_tool_use(...)      -> text that makes a CrewAI agent use a tool

Author: Ravi Shanker Singh (TechWithRavi) - https://github.com/replyravi
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables.config import get_config_list
from pydantic import BaseModel, Field, PrivateAttr


# ---------------------------------------------------------------------------
# 1. LangChain fake chat model
# ---------------------------------------------------------------------------
class FakeChatModel(GenericFakeChatModel):
    """A pretend chat model that answers with the replies you give it, in order.

    It works everywhere a real LangChain chat model works:
        model.invoke(...), model.stream(...), prompt | model | parser,
        model.bind_tools([...]), create_agent(model, tools=[...]),
        model.with_structured_output(MySchema)

    It also remembers every prompt it received in `.received`, so a test
    can check what your code actually sent to the "AI".
    """

    received: list = Field(default_factory=list)

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any):
        # A real model sends the list of tools to the AI provider.
        # Our fake already knows its scripted replies, so it just returns itself.
        return self

    def batch(self, inputs, config=None, **kwargs):
        # The scripted replies are handed out in order, so we answer a batch ONE input
        # at a time. (In parallel, a reply could be matched to the wrong input.)
        return super().batch(inputs, _one_at_a_time(config, len(inputs)), **kwargs)

    async def abatch(self, inputs, config=None, **kwargs):
        return await super().abatch(inputs, _one_at_a_time(config, len(inputs)), **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Remember what was sent to the model (useful in tests).
        self.received.append(messages)
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except StopIteration:
            raise RuntimeError(
                "FakeChatModel ran out of scripted replies. "
                "Add one more reply to fake_llm(...) in your test."
            ) from None

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        # Streaming = sending the answer in small pieces ("chunks").
        message = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs).generations[0].message

        if message.tool_calls:
            # A tool request is sent as ONE chunk (real models split it, tests don't care).
            tool_chunks = [
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i, "type": "tool_call_chunk"}
                for i, c in enumerate(message.tool_calls)
            ]
            chunk = AIMessageChunk(content="", tool_call_chunks=tool_chunks, chunk_position="last")
            yield ChatGenerationChunk(message=chunk)
            return

        # Plain text is sent word by word, like a real model.
        # (message.text works for plain text AND for Gemini-style lists of content blocks)
        words = re.split(r"(\s)", message.text)
        for i, word in enumerate(words):
            chunk = AIMessageChunk(content=word)
            if i == len(words) - 1:
                chunk.chunk_position = "last"
            generation_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(word, chunk=generation_chunk)
            yield generation_chunk


def _one_at_a_time(config, count: int) -> list:
    """Copy the batch config(s) and set max_concurrency=1 (used by FakeChatModel.batch)."""
    return [{**c, "max_concurrency": 1} for c in get_config_list(config, count)]


def fake_llm(*replies: str | AIMessage) -> FakeChatModel:
    """Make a fake chat model.

    Each time the model is called it uses the NEXT reply:
        llm = fake_llm("Hello!", "Second answer")
        llm.invoke("hi").content   -> "Hello!"
        llm.invoke("again").content -> "Second answer"

    A reply can be plain text, or an AIMessage made with tool_call(...)
    or structured_reply(...).
    """
    return FakeChatModel(messages=iter(list(replies)))


def tool_call(name: str, args: dict, call_id: str | None = None) -> AIMessage:
    """A scripted reply where the model asks to run the tool `name` with `args`.

    Example (the agent will call the `add` tool with a=2, b=3):
        tool_call("add", {"a": 2, "b": 3})
    """
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": call_id or f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }
        ],
    )


def structured_reply(answer: BaseModel) -> AIMessage:
    """A scripted reply for `llm.with_structured_output(MySchema)`.

    with_structured_output asks the model to "call a tool" whose name is the
    schema's class name, so we script exactly that tool call.

    Example:
        llm = fake_llm(structured_reply(Movie(title="Inception", year=2010)))
        llm.with_structured_output(Movie).invoke("...")  -> Movie(title='Inception', year=2010)
    """
    return tool_call(type(answer).__name__, answer.model_dump())


def gemini_style_reply(text: str) -> AIMessage:
    """A scripted reply shaped like a real Gemini 3+ answer.

    Gemini 3+ models return message.content as a LIST of blocks, not a string:
        [{"type": "text", "text": "Hello", "extras": {"signature": "..."}}]
    So always read answers with message.text (or StrOutputParser), never message.content.
    Use this in a test to make sure your code does that.
    """
    return AIMessage(content=[{"type": "text", "text": text, "extras": {"signature": "fake-signature"}}])


def fake_embeddings(size: int = 384) -> DeterministicFakeEmbedding:
    """Offline embeddings: the same text always gives the same vector.

    Good for testing that your vector store / RAG code RUNS.
    (The vectors are random-looking, so similarity is not meaningful.
    For meaningful search offline, use the real local model from llm.get_embeddings().)
    """
    return DeterministicFakeEmbedding(size=size)


# ---------------------------------------------------------------------------
# 2. CrewAI fake LLM (only defined when CrewAI is installed)
# ---------------------------------------------------------------------------
try:
    from crewai import BaseLLM as _CrewAIBaseLLM
except ImportError:  # CrewAI is only installed for the CrewAI episodes.
    _CrewAIBaseLLM = None


def react_final_answer(answer: str, thought: str = "I now know the final answer") -> str:
    """Text that makes a CrewAI agent finish its task with `answer`."""
    return f"Thought: {thought}\nFinal Answer: {answer}"


def react_tool_use(tool_name: str, tool_input: dict, thought: str = "I should use a tool") -> str:
    """Text that makes a CrewAI agent run the tool `tool_name` with `tool_input`."""
    return f"Thought: {thought}\nAction: {tool_name}\nAction Input: {json.dumps(tool_input)}"


if _CrewAIBaseLLM is not None:

    class FakeCrewAILLM(_CrewAIBaseLLM):
        """A pretend LLM for CrewAI agents. Replies with `responses`, in order.

        CrewAI agents that have no native tool calling talk in a simple text
        format ("Thought / Action / Action Input / Final Answer"). Use
        react_final_answer(...) and react_tool_use(...) to write the replies.

            llm = FakeCrewAILLM(responses=[react_final_answer("Hi!")])
            agent = Agent(role="...", goal="...", backstory="...", llm=llm)
        """

        responses: list[str] = Field(default_factory=list)
        received: list = Field(default_factory=list)
        _next_index: int = PrivateAttr(default=0)

        def __init__(self, responses: list[str] | None = None, **kwargs):
            # CrewAI insists that every LLM has a model name, so we give it one.
            kwargs.setdefault("model", "fake-crewai-llm")
            super().__init__(responses=responses or [], **kwargs)

        def call(
            self,
            messages,
            tools=None,
            callbacks=None,
            available_functions=None,
            from_task=None,
            from_agent=None,
            response_model=None,
            **kwargs,
        ):
            # Remember what CrewAI sent (useful in tests).
            self.received.append(messages)
            if self._next_index >= len(self.responses):
                raise RuntimeError(
                    "FakeCrewAILLM ran out of scripted responses. "
                    "Add one more item to responses=[...] in your test."
                )
            reply = self.responses[self._next_index]
            self._next_index += 1
            return reply

        def supports_function_calling(self) -> bool:
            # False -> CrewAI uses the text format (Thought/Action/Final Answer).
            return False

        def supports_stop_words(self) -> bool:
            return False

        def get_context_window_size(self) -> int:
            return 8192
