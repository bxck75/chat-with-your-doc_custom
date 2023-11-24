import os
import openai
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import (UnstructuredMarkdownLoader, BSHTMLLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader, PyPDFLoader, UnstructuredFileLoader,PythonLoader, CSVLoader, MWDumpLoader)
import langchain.text_splitter as text_splitter
from langchain.text_splitter import (RecursiveCharacterTextSplitter, CharacterTextSplitter)

from typing import List
import streamlit
import glob

REQUEST_TIMEOUT_DEFAULT = 10
TEMPERATURE_DEFAULT = 0.0
CHAT_MODEL_NAME_DEFAULT = "gpt-3.5-turbo"
OPENAI_EMBEDDING_DEPLOYMENT_NAME_DEFAULT = "text-embedding-ada-002"
CHUNK_SIZE_DEFAULT = 1000
CHUNK_OVERLAP_DEFAULT = 0

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


import os
import sys
import textwrap

from dotenv import load_dotenv, find_dotenv
from typing import List, Optional
from huggingface_hub import login
from langchain.llms import HuggingFaceHub
from langchain.chains import  LLMChain
from elevenlabs import set_api_key
import pandas as pd
from langchain.tools import tool
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain.chains import LLMMathChain,LLMChain
from langchain.schema import SystemMessage
from langchain.document_loaders import YoutubeLoader
from langchain_experimental.autonomous_agents import HuggingGPT
from langchain.chains.summarize import load_summarize_chain
from langchain.utilities import PythonREPL
from langchain.utilities.serpapi import SerpAPIWrapper
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.tools import ElevenLabsText2SpeechTool
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    Language,
)
from langchain.prompts import load_prompt, ChatPromptTemplate, MessagesPlaceholder
from transformers import load_tool
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import load_prompt
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import WebBaseLoader
from tempfile import TemporaryDirectory
from langchain.agents.agent_toolkits import FileManagementToolkit
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
import getpass
import os
import sys

import textwrap
import pandas as pd
import faiss
load_dotenv(find_dotenv())
from credits import (
    HUGGINGFACE_TOKEN,
    OPENAI_API_KEY,
    HUGGINGFACE_TOKEN as HUGGINGFACEHUB_API_TOKEN,
    HUGGINGFACE_EMAIL,
    HUGGINGFACE_PASS,
    ELEVENLABS_API_KEY,
    SERPAPI_API_KEY)

serp_search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
tts = ElevenLabsText2SpeechTool(eleven_api_key=ELEVENLABS_API_KEY, voice="amy",)
script_dir = os.path.dirname(os.path.abspath(__file__))
two_folders_up = os.path.abspath(os.path.join(script_dir, '..', '..'))
working_directory = TemporaryDirectory()
sys.path.append(two_folders_up)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["ELEVENLABS_API_KEY"] = ELEVENLABS_API_KEY
os.environ["CHAT_MODEL_NAME"]="gpt-3.5-turbo"
set_api_key(ELEVENLABS_API_KEY)
login(HUGGINGFACEHUB_API_TOKEN)

class DocChatbot:
    ''' 
    Here's the format for chat history:
    [{"role": "assistant", "content": "How can I help you?"}, 
    {"role": "user", "content": "What is your name?"}]
    The input for the Chain is in a format like this:
    [("How can I help you?", "What is your name?")]
    That is, it's a list of question and answer pairs.
    So need to transform the chat history to the format for the Chain
    '''  

    llm: HuggingFaceHub
    tools: List[Tool]
    embeddings: HuggingFaceEmbeddings
    vector_db: FAISS
    chunk_size = int
    chunk_overlap = int
    chatchain: BaseConversationalRetrievalChain
    request_timeout: int
    temperature: float
    chat_model_name : str
    api_key : str

    def __init__(self) -> None:
        self.embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.chunk_size = 512
        self.temperature = 0.1
        self.chunk_overlap = 8 
        self.request_timeout =30 
        self.max_new_tokens = 500
        
        self.llm = HuggingFaceHub( huggingfacehub_api_token=HUGGINGFACE_TOKEN,repo_id="tiiuae/falcon-7b-instruct", model_kwargs={ "temperature": self.temperature ,"max_new_tokens": self.max_new_tokens })
        self.embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
        # tools
        self.tools=[
            Tool.from_function(
                name="Search",
                func=SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY).run,
                description="Useful for searching online and enhancing your response. Input is your question",
            ),
            Tool.from_function(
                name="Painter",
                func=self.text2image,
                description=
                '''Useful for when you want to compliment your response with images. 
                    The inputs are:
                    (1) a detailed description of the image 
                    (2) a filename so it can be stored''',
                return_direct=False, 
            ),
            Tool.from_function(
                name="Speak",
                func=tts.stream_speech,
                description="Useful for when you want to enhance your response with your voice",
                return_direct=False,   
            ),
        ]
        self.tools.append(self.text2image)
        # list of tool names
        self.tool_names = [tool.name for tool in self.tools]  

    @tool('text2image')
    def text2image(prompt, file):
        """
        Convert a text prompt into an image and save it to a specified file.

        Parameters:
            prompt (str): The text prompt to convert into an image.
            file (str): The path to the file where the resulting image will be saved.

        Returns:
            None: The function saves the resulting image to a file and does not return anything.

        Examples:
            text2image("Draw a blue square", "blue_square.png")
        """
        from huggingface_hub import InferenceClient
        client = InferenceClient()
        image = client.text_to_image(prompt)
        image.save(file)
            
    def init_chatchain(self, chain_type : str = "stuff") -> None:
        #ConversationalRetrievalChain prompt
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
            Given the following conversation and a follow up input, 
            rephrase and speak out the standalone question. 
            The standanlone question to be generated should be in the same language with the input. 
            You can gather more information, answer with your voice and generate an image from text by using 
            Always try to use your ability to speak when rephrasing or answering. 
            the tools:
            {tool_names}
           
            Chat History:
            {chat_history}
                                                                
            Follow Up Input:
            {question}
                                                                
            Standalone Question:
            """)
    
        # stuff chain_type seems working better than others
        self.chatchain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(),
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            chain_type=chain_type,
            return_source_documents=True,
            verbose=True,
            #combine_docs_chain_kwargs=dict(return_map_steps=True)
        )

    # get answer and source documents from query, 
    def get_answer_with_source(self, tool_names, query, chat_history):
        result = self.chatchain({   "tool_names":self.tool_names, 
                                    "question": query, 
                                    "chat_history": chat_history
                            },return_only_outputs=True)
        return result['answer'], result['source_documents']

    # get answer from query. 
    def get_answer(self, tool_names, query, chat_history):
        chat_history_for_chain = []
        
        for i in range(0, len(chat_history), 2):
            chat_history_for_chain.append(( chat_history[i]['content'], chat_history[i+1]['content'] 
                    if chat_history[i+1] is not None else "" )
            )
        result = self.chatchain({"tool_names":self.tool_names, 
                                    "question": query, 
                                    "chat_history": chat_history
                                        },return_only_outputs=True)
        return result['answer'], result['source_documents']
        
    # load vector db from local
    def load_vector_db_from_local(self, path: str, index_name: str):
        self.vector_db = FAISS.load_local(path, self.embeddings, index_name)
        #self.vector_db = Chroma(persist_directory=path, embedding_function=self.embeddings)
        print(f"Loaded vector db from local: {path}/{index_name}")

    # save vector db to local
    def save_vector_db_to_local(self, path: str, index_name: str):
        #self.vector_db.persist(persist_directory=path)
        FAISS.save_local(self.vector_db, path, index_name)
        print("Vector db saved to local")

    # split, embed and ingest
    def init_vector_db_from_documents(self, file_list: List[str]):
        from langchain.text_splitter import RecursiveCharacterTextSplitter, PythonCodeTextSplitter, MarkdownTextSplitter, TextSplitter
        from langchain.document_loaders import UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader, PyPDFLoader, CSVLoader, PythonLoader, MWDumpLoader, UnstructuredMarkdownLoader, BSHTMLLoader
        import os

        text_splitter = RecursiveCharacterTextSplitter()
        python_splitter = PythonCodeTextSplitter() 
        md_splitter = MarkdownTextSplitter()
       

        docs = []
        for file in file_list:
            print(f"Loading file: {file}")
            ext_name = os.path.splitext(file)[-1]
            
            if ext_name == ".pptx":
                loader = UnstructuredPowerPointLoader(file)
            elif ext_name == ".docx":
                loader = UnstructuredWordDocumentLoader(file)
            elif ext_name == ".pdf":
                loader = PyPDFLoader(file)
            elif ext_name == ".csv":
                loader = CSVLoader(file_path=file) 
            elif ext_name == ".py":
                loader = PythonLoader(file_path=file)
            elif ext_name == ".xml":
                loader = MWDumpLoader(file_path=file, encoding="utf8")
            elif ext_name == ".md":
                loader = UnstructuredMarkdownLoader(file, mode="elements")
            elif ext_name == ".html":
                loader = BSHTMLLoader(file)
            else:
                raise ValueError(f"Unsupported extension: {ext_name}")
                
            loaded_docs = loader.load()
            
            if ext_name in [".pptx", ".docx", ".pdf", ".xml"]:
                split_docs = text_splitter.split_documents(loaded_docs) 
            elif ext_name == ".py":
                split_docs = python_splitter.split_documents(loaded_docs)
            elif ext_name == ".md":
                split_docs = md_splitter.split_documents(loaded_docs) 
            elif ext_name == ".html":
                split_docs = text_splitter.split_documents(loaded_docs)
                
            docs.extend(split_docs)
                
        print(f"Loaded and split {len(docs)} documents")
        print("Generating embeddings and ingesting.")
        self.vector_db = FAISS.from_documents(docs, self.embeddings)
        #self.vector_db = Chroma.from_documents(
         #       docs, 
         #       self.embeddings, 
         #       persist_directory=path
         #   )
        print("Vector db initialized.")

    # Get indexes
    def get_available_indexes(self, path: str):
        return [os.path.splitext(os.path.basename(file))[0] for file in glob.glob(f"{path}/*.faiss")]
        