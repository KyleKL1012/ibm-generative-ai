import os
import PyPDF2
import re
from dotenv import load_dotenv

from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai.credentials import Credentials

# Load environment variables
load_dotenv()
api_key = os.getenv("GENAI_KEY", None)
api_url = os.getenv("GENAI_API", None)
creds = Credentials(api_key, api_endpoint=api_url)

# Load the PDF file and extract text
pdf_path = "examples/POC datasets/Car Parts Sales Contract1.pdf"
with open(pdf_path, "rb") as file:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

# Find the discount information in the extracted text
discount = None
discount_match = re.search(r"(\d+(?:\.\d+)?)%", text)
if discount_match:
    discount = float(discount_match.group(1))

print(f"The discount mentioned in the document is: {discount}%")

# Use GenAI LangChain model to answer questions based on the extracted text
print("\n------------- Example (LangChain)-------------\n")

params = GenerateParams(decoding_method="greedy")

print("Using GenAI Model expressed as LangChain Model via LangChainInterface:")

langchain_model = LangChainInterface(model="google/flan-t5-xxl", params=params, credentials=creds)
question = "Who is Joy mentioned in the document? The context is: " + text
answer = langchain_model(question)

print(f"Answer: {answer}")