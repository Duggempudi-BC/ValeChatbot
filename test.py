# import xorbits.pandas as pd

# data = pd.read_excel("ValeChatbottextfile (1).xlsx")
# print(data)


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import tempfile
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')
uploaded_file = "ValeChatbottextfile (1).csv"

if uploaded_file :
   #use tempfile because CSVLoader only accepts a file_path
    # with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
    #     tmp_file.write(uploaded_file.getvalue())
    #     tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path="ValeChatbottextfile (1).csv", encoding="utf-8", csv_args={
                'delimiter': ','})
    data = loader.load()
print(data)

embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_endpoint= os.getenv('AZURE_OPENAI_ENDPOINT'),
    openai_api_key= os.getenv('AZURE_OPENAI_API_KEY'),
    openai_api_version='2023-09-01-preview'
)
vectorstore = FAISS.from_documents(data, embeddings)
vectorstore.save_local("chatbot_index")

