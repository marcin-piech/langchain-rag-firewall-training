# Prompt Template Docs:
#   https://python.langchain.com/v0.2/docs/concepts/#prompt-templateshttps://python.langchain.com/v0.2/docs/concepts/#prompt-templates

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# # PART 1: Create a ChatPromptTemplate using a template 
# template = "Tell me {num_of_facts} intersting facts about {country_name}."
# prompt_template = ChatPromptTemplate.from_template(template)

# print("----- Prompt from Template -----")
# prompt = prompt_template.invoke({"num_of_facts": "3",
#                                    "country_name": "Switzerland"})
# print(prompt)


# PART 2: Prompt with System and Human Messages (Using 
messages = [
  ("system", "You are a person who lives in {country}. You can tell interesting facts about the country."),
  ("human", "Tell me {num_of_facts} interesting facts.")
            ]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"country": "Switzerland", "num_of_facts": 3})
print("\n---------- Prompt with System and Human Messages (Tuple) ---------\n")
print(prompt)



# # Extra Informoation about Part 2.
# This does work:
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me 3 jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers"})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)


# This does NOT work:
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)
