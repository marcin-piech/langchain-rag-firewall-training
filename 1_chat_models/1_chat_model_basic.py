# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/openai/

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# my code
load_dotenv()

model = ChatOpenAI(model="gpt-4o")
result = model.invoke("Hi, who invented a dynomite?")
print("ChatGPT:\n")
print(result )
print("\n\nContent only:\n")
print(f"{result.content}" + "\n")
