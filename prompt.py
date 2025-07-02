from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages

def get_prompt_template():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are JOYI, a helpful and intelligent female assistant. Answer all questions in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

def get_trimmer(model):
    return trim_messages(
        max_tokens=65,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

prompt_template = get_prompt_template()
trimmer = None  # initialized in main with model context
