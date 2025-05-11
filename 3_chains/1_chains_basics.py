from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo-0125")

prompt_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "tell me {num_of_jokes} jokes."),
  ]
)

chain = prompt_template | model | StrOutputParser()

result = chain.invoke({"topic": "students", "num_of_jokes": 1})

print(result)