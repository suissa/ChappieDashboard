"""Shared State feature."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint, AGUIToolset
from dotenv import load_dotenv
from fastapi import FastAPI
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()


class ProverbsState(BaseModel):
    """List of the proverbs being written."""

    proverbs: list[str] = Field(
        default_factory=list,
        description="The list of already written proverbs",
    )


def set_proverbs(tool_context: ToolContext, new_proverbs: list[str]) -> Dict[str, str]:
    """Set the list of proverbs using the provided new list."""
    try:
        tool_context.state["proverbs"] = new_proverbs
        return {"status": "success", "message": "Proverbs updated successfully"}
    except Exception as e:
        return {"status": "error", "message": f"Error updating proverbs: {str(e)}"}


def get_weather(tool_context: ToolContext, location: str) -> Dict[str, str]:
    """Get the weather for a given location."""
    return {"status": "success", "message": f"The weather in {location} is sunny."}


def on_before_agent(callback_context: CallbackContext):
    """Initialize proverbs state if it doesn't exist."""
    if "proverbs" not in callback_context.state:
        callback_context.state["proverbs"] = []
    return None


def before_model_modifier(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Add the current proverb state to the model instruction."""
    if callback_context.agent_name == "ProverbsAgent":
        proverbs_json = "No proverbs yet"
        if callback_context.state.get("proverbs") is not None:
            try:
                proverbs_json = json.dumps(callback_context.state["proverbs"], indent=2)
            except Exception as e:
                proverbs_json = f"Error serializing proverbs: {str(e)}"
        original_instruction = llm_request.config.system_instruction or types.Content(
            role="system", parts=[]
        )
        prefix = f"""You are a helpful assistant for maintaining a list of proverbs.
This is the current state of the list of proverbs: {proverbs_json}
When you modify the list of proverbs, use the set_proverbs tool to update the list."""
        if not isinstance(original_instruction, types.Content):
            original_instruction = types.Content(
                role="system", parts=[types.Part(text=str(original_instruction))]
            )
        if not original_instruction.parts:
            original_instruction.parts = [types.Part(text="")]
        original_instruction.parts[0].text = (
            prefix + (original_instruction.parts[0].text or "")
        )
        llm_request.config.system_instruction = original_instruction
    return None


def simple_after_model_modifier(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Stop the consecutive tool calling of the agent."""
    if callback_context.agent_name == "ProverbsAgent":
        if llm_response.content and llm_response.content.parts:
            if (
                llm_response.content.role == "model"
                and llm_response.content.parts[0].text
            ):
                callback_context._invocation_context.end_invocation = True
    return None


# The scheduled/PR smoke tests run against aimock. ADK's native Gemini client
# talks directly to Google and cannot be redirected to aimock, so CI switches
# to ADK's LiteLLM adapter and the OpenAI-compatible aimock endpoint. Production
# and local development keep the real Gemini model by default.
smoke_model = (
    LiteLlm(model="openai/gpt-4o-mini")
    if os.getenv("COPILOTKIT_SMOKE_USE_AIMOCK") == "1"
    else "gemini-2.5-flash"
)

proverbs_agent = LlmAgent(
    name="ProverbsAgent",
    model=smoke_model,
    instruction="""
        When a user asks you to do anything regarding proverbs, you MUST use the set_proverbs tool.
        Always pass the COMPLETE LIST of proverbs to the set_proverbs tool.
        After using the tool, provide a brief summary.
        Only call get_weather when the user asks for weather in a location.
    """,
    tools=[set_proverbs, get_weather, AGUIToolset()],
    before_agent_callback=on_before_agent,
    before_model_callback=before_model_modifier,
    after_model_callback=simple_after_model_modifier,
)

adk_proverbs_agent = ADKAgent(
    adk_agent=proverbs_agent,
    user_id="demo_user",
    session_timeout_seconds=3600,
    use_in_memory_services=True,
)

app = FastAPI(title="ADK Middleware Proverbs Agent")
add_adk_fastapi_endpoint(app, adk_proverbs_agent, path="/")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
