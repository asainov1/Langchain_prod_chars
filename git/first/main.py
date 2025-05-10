import os
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
#openai key
os.environ['OPENAI_API_KEY'] = ''
#langchain key
os.environ['LANGCHAIN_TRACING_V2'] = ''
os.environ['LANGCHAIN_API_KEY'] = ''