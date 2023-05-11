from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain

import textwrap

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=BWJ4vnXIvts", add_video_info=True)

result = loader.load()

print (type(result))
print (f"Found video from {result[0].metadata['author']} that is {result[0].metadata['length']} seconds long")
print ("")
print (result)

chain = load_summarize_chain(llm, chain_type="stuff", verbose=False)
chain.run(result)
