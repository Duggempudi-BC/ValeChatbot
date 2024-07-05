from langchain_community.document_loaders import UnstructuredExcelLoader, UnstructuredWordDocumentLoader
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
import os
from typing import List
from dotenv import load_dotenv
import asyncio
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.utilities.sql_database import SQLDatabase
import io
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import AzureChatOpenAI

load_dotenv(dotenv_path='.env')

async def main():
#     loader = UnstructuredExcelLoader("ValeChatbottextfile.xlsx", mode="single")
#     docs = await loader.aload()
#     print(len(docs))
#     # print(type(docs))
#     docs = [Document(page_content=docs[0].metadata['text_as_html'])]
#     # loader = UnstructuredWordDocumentLoader("Finalcampaigndata.docx")
#     md = MarkdownifyTransformer()
#     docs = md.transform_documents(docs)
#     # print(len(docs))
#     # # docs = loader.load()
#     # print(docs)
#     # print(len(docs))
#     text_splitter = CharacterTextSplitter(
#     separator="|\\n|",
#     length_function=len,
#     is_separator_regex=False,
# )
#     texts = text_splitter.create_documents([docs[0].page_content])
#     print(texts)
#     print(len(texts))
    # all_texts = []
    # with open("Finalcampaigndata.docx", "rb") as uploaded_file:
    #         file_contents = uploaded_file.read()
    #         # Create an in-memory buffer from the file content
    #         bytes = io.BytesIO(file_contents)

    #         doc = Document(bytes)
    #         paragraph_list = []
    #         for paragraph in doc.paragraphs:
    #             paragraph_list.append(paragraph.text)
    #             text = "\n".join(paragraph_list)

    #             # Split the text into chunks
    #         text_splitter = RecursiveCharacterTextSplitter(
    #             chunk_size=1000,
    #             chunk_overlap=10,
    #         )
    #         texts = text_splitter.split_text(text)

    #         # Add the chunks and metadata to the list
    #         all_texts.extend(texts)

    # Create a metadata for each chunk
    # metadatas = [{"source": f"{i}-pl"} for i in range(len(all_texts))]
    # embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    #     azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    #     openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    #     openai_api_version='2023-09-01-preview'
    # )
    # # db = Chroma(persist_directory='./chroma_db', embedding_function=embeddings, collection_name='excel_docs')
    # db = (Chroma.from_texts)(
    #     all_texts, embeddings, metadatas=metadatas
    # )
    # await db.aadd_documents(documents=data)
    # print('done')

    llm: AzureChatOpenAI = AzureChatOpenAI(
        azure_deployment="gpt-4-1106",
        openai_api_version="2023-09-01-preview",
    )
    



    db = SQLDatabase.from_uri(f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
    agent_executor.invoke(
    "what is the format for BM-D1-Leaderboard?"
)

if __name__=='__main__':
    asyncio.run(main())


