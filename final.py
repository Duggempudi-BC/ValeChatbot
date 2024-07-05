import uuid
from typing import Any, List

import chainlit as cl
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableParallel, RunnableSerializable
from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_experimental.agents.agent_toolkits import create_xorbits_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import FAISS
# from langchain_community.agent_toolkits import create_sql_agent
from operator import itemgetter
import os
from dotenv import load_dotenv


load_dotenv(dotenv_path='.env')

EMBEDDINGS: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    openai_api_version='2023-09-01-preview'
)

# NAMESPACE: str = "pgvector/cbot_corpus"
# COLLECTION_NAME: str = 'CBOT-CORPUS'
db_config = {
    'user': 'ahsannayaz',
    'password': 'Qtnkwvnv7632!',
    'host': 'jirabot-pg-storage.postgres.database.azure.com',
    'port': '5432',
    'database': 'chatbot-corpus'
}


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Ratecard for BM-S1-Content Hub",
            message="Please give me the ratecard for BM-S1-Content Hub",
            # icon="/public/idea.svg",
            ),

        cl.Starter(
            label="Description for BM-S8-FA-Newsletter Category Sponsorship-Financial Analysis",
            message="Please give me the description for BM-S8-FA-Newsletter Category Sponsorship-Financial Analysis",
            # icon="/public/learn.svg",
            ),
        cl.Starter(
            label="Format for LB-1- Live Broadcast",
            message="What is the format for LB-1- Live Broadcast",
            # icon="/public/terminal.svg",
            ),
        cl.Starter(
            label="Units sold for BM-S5-Event Promotion",
            message="Please give me units sold for BM-S5-Event Promotion",
            # icon="/public/write.svg",
            )
        ]


@cl.on_chat_start
async def chat_start():
    llm: AzureChatOpenAI = AzureChatOpenAI(
        azure_deployment="gpt-4-32k",
        openai_api_version="2023-09-01-preview",
    )
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
'''
<history>\n{history}\n</history>

You are a helpful assistant, utilize the given context to answer the user query

Rules:
Your primary task is to answer questions based STRICTLY on the provided context.
If the question does NOT directly match with the context, respond with  I don't know.
Always use more text to elaborate the answer. However, ensure the elaboration is strictly based on the context.

Remember: Stick to the context. If uncertain, respond with I don't know.

<context>
{context}
</context>


<question>\n{input}\n</question>

        '''
    )
    vectorstore = FAISS.load_local("chatbot_index", EMBEDDINGS, allow_dangerous_deserialization=True)
    memory: InMemoryChatMessageHistory = InMemoryChatMessageHistory()
    retriever: VectorStoreRetriever = vectorstore.as_retriever()
    chain: RunnableSerializable[Any, BaseMessage] = (
            {
                "context": itemgetter("input") | retriever,
                "input": itemgetter("input"),
                "history": itemgetter("history"),
            }
            | prompt
            | llm
    )
    # db = SQLDatabase.from_uri(f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
    # agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", prompt=prompt, verbose=True)
    # data = pd.read_excel("ValeChatbottextfile (1).xlsx")
    # agent_executor = create_xorbits_agent(llm=llm, data=data, verbose=True, allow_dangerous_code=True)
    # cl.user_session.set('runnable_history', runnable_initial)
    cl.user_session.set('memory', memory)
    cl.user_session.set('llm', llm)
    cl.user_session.set('runnable', chain)


@cl.on_message
async def on_message(message: cl.Message):
    runnable: RunnableSerializable[Any, Any] = cl.user_session.get('runnable')
    memory: InMemoryChatMessageHistory = cl.user_session.get('memory')
    user_id: uuid.uuid4 = cl.user_session.get("id")
    actions = cl.user_session.get('actions')
    if actions:
        for action in actions:
            await action.remove()
    actions = [
        cl.Action(name="Summary", value="Summary", description="Click me for summary!"),
        cl.Action(name="Sentiment Analysis", value="Sentiment Analysis", description="Click me for Sentiment Analysis!")
    ]
    cl.user_session.set('actions', actions)
    msg: cl.Message = cl.Message(content="", actions=actions)
    async for chunk in runnable.astream(
            {"input": message.content, "history": memory.messages},
            config=RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler()])
    ):
        print(chunk)
        await msg.stream_token(token=chunk.content)
    await msg.send()
    await memory.aadd_messages([HumanMessage(content=message.content),
                                AIMessage(content=msg.content)])
    
import chainlit as cl

@cl.action_callback("Summary")
async def on_action(action):
    memory: InMemoryChatMessageHistory = cl.user_session.get('memory')
    llm = cl.user_session.get('llm')
    res = llm.invoke(f'''Summarise the following conversation
               <conversation>
               {memory.messages}
                </conversation>
               ''')
    await cl.Message(content=f'### Summary: \n{res.content}').send()
    # Optionally remove the action button from the chatbot user interface
    await action.remove()


@cl.action_callback("Sentiment Analysis")
async def on_action(action):
    """
Perform Sentiment Analysis on the given conversation and respond with Positive, Negative or Neutral. 
To make your decision, include all synonyms for negative, positive and neutral messages.

Parameters:
    action (Any): The action triggering the sentiment analysis.

Returns:
    None
"""
    memory: InMemoryChatMessageHistory = cl.user_session.get('memory')
    llm = cl.user_session.get('llm')
    res = llm.invoke(f'''Perform Sentiment Analysis on the given conversation and respond with Positive, Negative or Neutral. To make your decision, include all synonyms for negative, positive and neutral messaged.

                     
               <conversation>
               {memory.messages}
                </conversation>
               ''')
    await cl.Message(content=f'### Sentiment: \n{res.content}').send()
    # Optionally remove the action button from the chatbot user interface
    await action.remove()

 