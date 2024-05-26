from dotenv import load_dotenv
load_dotenv()

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage

import warnings
warnings.filterwarnings("ignore")

def get_model_list():
    return ChatNVIDIA.get_available_models()

def get_llm(model):
    llm=ChatNVIDIA(model=model)
    return llm

def get_llm_response(human_message,model="ai-mistral-7b-instruct-v2"):
    messages=[
        SystemMessage(content="You are chatbot assistant"),
        HumanMessage(content=human_message)
    ]
    llm=get_llm(model)
    return llm.invoke(messages).content



# print(get_model_list())

