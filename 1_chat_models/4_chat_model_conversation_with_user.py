from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo-0125")

chat_history = []

system_message = SystemMessage(content="You are a helpful AI asistant.")
chat_history.append(system_message)

while True:
  query = input("You: ")
  if query.lower() in ("exit", "q", "quit", "bye", "goodbye"):
    break
  chat_history.append(HumanMessage(content=query))

  result = model.invoke(chat_history)
  response = result.content
  chat_history.append(AIMessage(content=response))
  print(f"ChatGPT: {response}\n")


print("-------- Message History --------")
print(chat_history)