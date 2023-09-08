import os
from dotenv import load_dotenv

from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai.credentials import Credentials
from genai.model import Model

from langchain import PromptTemplate, LLMChain, HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

load_dotenv()
api_key = os.getenv("GENAI_KEY", None)
api_url = os.getenv("GENAI_API", None)
creds = Credentials(api_key, api_endpoint=api_url)

print("\n------------- Example (LangChain)-------------\n")

# Instantiate parameters for text generation
params = GenerateParams(decoding_method="sample", max_new_tokens=100)

print("Using GenAI Model expressed as LangChain Model via LangChainInterface:")

# Instantiate a model proxy object to send your requests
flan_ul2 = LangChainInterface(model="google/flan-ul2", params=params, credentials=creds)

chat_history =[]
topic = 'Text2SQL'

template = """
        You are a SQL export to generate SQL query based on Table Schema below.

        Table "department" Schema: CREATE TABLE IF NOT EXISTS "department" (
                            "Department_ID" int,
                            "Name" text,
                            "Creation" text,
                            "Ranking" int,
                            "Budget_in_Billions" real,
                            "Num_Employees" real,
                            PRIMARY KEY ("Department_ID")
                        );

        Here are some examples:
        Question: How many heads of the departments are older than 56 ?
        Query: SELECT count(*) FROM head WHERE age  >  56

        Question: List the name, born state and age of the heads of departments ordered by age.
        Query: SELECT name ,  born_state ,  age FROM head ORDER BY age

        Question:{question}
        Query:
    """
question = "List the creation year, name and budget of each department."

prompt = PromptTemplate(template=template, input_variables=['question'])
llm_chain = LLMChain(prompt=prompt, llm=flan_ul2)
result = llm_chain.run(question)


print("Chatbot: ",result)


