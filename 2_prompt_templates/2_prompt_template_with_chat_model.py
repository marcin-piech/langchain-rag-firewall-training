from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo-0125")

# PART 1: Create a ChatPromptTemplate using a template string
print("--------- Prompt from Template ---------")
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)
prompt = prompt_template.invoke({"topic": "job"})
result = model.invoke(prompt)
print(result.content)


# PART 2: Prompt with System and Human Messages (Using Tuples)
print("--------- Prompt from multiple Template (Tuples) ---------")
messages = [
  ("system", "You are a funny comedian who tells jokes about {topic}"),
  ("human", "Tell me {num_of_jokes} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "job", "num_of_jokes": 2})
result = model.invoke(prompt)
print(result.content)
