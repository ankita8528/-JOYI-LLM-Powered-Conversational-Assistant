from langgraph.graph import START, StateGraph
from langchain_core.messages import AIMessage
from typing import Sequence
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

class State(TypedDict):
    messages: Annotated[Sequence, add_messages]
    language: str

def get_workflow(model, prompt_template, trimmer):
    def call_model(state: State):
        # Trim previous messages
        trimmed = trimmer.invoke(state["messages"])

        # Create prompt based on trimmed history and language
        prompt = prompt_template.invoke({
            "messages": trimmed,
            "language": state["language"]
        })

        # Generate full response by consuming model stream
        full_content = ""
        for chunk in model.stream(prompt):
            token = getattr(chunk, "content", "")
            full_content += token

        # Return final message (no generator!)
        return {"messages": [AIMessage(content=full_content)]}

    # Define workflow graph
    workflow = StateGraph(state_schema=State)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    return workflow
