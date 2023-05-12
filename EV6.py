#pip install langchain openai faiss-cpu tiktoken tika

import openai
import os, json
from tika import parser
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import gradio as gr

# API Key값 가져오기
with open('./secret.json', 'r') as f:
    json_data = json.load(f)

API_KEY = json_data['openAI']

# OpenAI API key 등록
os.environ["OPENAI_API_KEY"] = API_KEY

# 경로 확인용
print(os.getcwd())

# PDF 파일 경로
pdf_path = "./PDF/EV6_total.pdf" 

# PDF 텍스트 추출
raw_pdf = parser.from_file(pdf_path)
contents = raw_pdf['content']
contents = contents.strip()

# 텍스트 추출 확인 
# print(contents)

# 텍스트 파일로 결과 저장
# file = open("contents.txt", "w")
# file.write(contents)
# file.close()

# Langchain 사용
loader = TextLoader('./contents.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Text Split 확인
# print(texts)

# embeddings 모듈 적용
embeddings = OpenAIEmbeddings()

# gpt-3.5-turbo 모델 적용
chat = ChatOpenAI(temperature=0)

# vector DB 생성
index = VectorstoreIndexCreator(
    vectorstore_cls=FAISS,
    embedding=embeddings,    
    ).from_loaders([loader])

# vector DB에 데이터 Save
# index.vectorstore.save_local("faiss_index")

# vector DB에 데이터 Load
index.vectorstore.load_local("faiss_index", embeddings)

# answer = index.query("EV6 트림은 뭐가 있어?", llm=chat, verbose=False)

# print(answer)
# print(type(answer))

# UI 구성
def add_text(history, text):
    print("text : " + text)
    history = history + [(text, None)]    
    return history, ""

def bot(history):
    response =index.query(history[-1][0], llm=chat, verbose=False)
    print("response : " + response)
    print(history)
    history[-1][1] = response
    return history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=750)

    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter",
            ).style(container=False)        

    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )    

demo.launch()
