from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo-0125")

cat_description_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful animal expert."),
    ("human", "Generate a short description related to the cat species described here: {description}")
  ]
)

bird_description_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful animal expert."),
    ("human", "Generate a short description related to the bird species described here: {description}")
  ]
)

fish_description_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful animal expert."),
    ("human", "Generate a short description related to the fish species described here: {description}")
  ]
)

default_description_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful animal expert."),
    ("human", "Generate a short general description of a type of animal related to the description here: {description}")
  ]
)

classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful animal expert."),
        ("human",
         "Classify the animal of this description as cat, bird, fish, or other (default): {feedback}."),
    ]
)

branches = RunnableBranch(
  (
    lambda x: "cat" in x,
    cat_description_template | model | StrOutputParser()
  ),
  (
    lambda x: "bird" in x,
    bird_description_template | model | StrOutputParser()
  ),
  (
    lambda x: "fish" in x,
    fish_description_template | model | StrOutputParser()
  ),

  default_description_template | model | StrOutputParser()
)

classification_chain = classification_template | model | StrOutputParser()
chain = classification_chain | branches

review = "fresh- and brackish-water fish of the family Cyprinidae, native to most of Europe and western Asia. Fish called roach can be any species of the genera Rutilus, Leucos and Hesperoleucus, depending on locality. The plural of the term is also roach."
result = chain.invoke({"feedback": review})

print(result)