from langchain.chat_models import init_chat_model

def get_model():
    return init_chat_model("gemini-2.0-flash", model_provider="google_genai")
