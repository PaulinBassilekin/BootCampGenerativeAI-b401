# import necessary libraries
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# load OpenAI API key from environment variables
openai_api_key = os.environ["OPENAI_API_KEY"]

# import OpenAI librairies from LangChain
from langchain_openai import OpenAI, ChatOpenAI

# initialize the OpenAI model for text completion
completionTextllmModel = OpenAI()

print("\n---------------\n")

#invoke text completion model with prompt
response = completionTextllmModel.invoke("Tell fun fact about paulin bassilekin")

print("==> Tell me fun fact about Paulin Bassilekin : Text Completion Response")

print(response)

print("\n---------------\n")

print("==> Streaming Text Completion Response")

for chunk in completionTextllmModel.stream("Tell fun fact about Paulin Bassilekin"):
    print(chunk, end="", flush=True)

# initialize gpt-4o-mini model for chat prompt
chatllmModel = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)

#Message for chat model
message = [
    ("system","your are a historian expert in the Bassilekin Paulin family."),
    ("human", "Tell me one fun fact about Paulin Bassilekin."),
    ("ai","sure!")
]

print("\n---------------\n")

print("==> Tell me fun fact about Paulin Bassilekin : Chat Model Response")
response = chatllmModel.invoke(message)

print(response.content)

print("\n---------------\n")

print("==> Streaming Chat Model Response")

for chunk in chatllmModel.stream(message):
    print(chunk.content, end="", flush=True)