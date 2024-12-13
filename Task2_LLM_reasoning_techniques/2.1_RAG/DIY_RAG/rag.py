from langchain.document_loaders import TextLoader     

loader = TextLoader("./data/the_bitter_lesson.txt")   
documents = loader.load()   

# 文档分割   
from langchain.text_splitter import CharacterTextSplitter      

# 创建拆分器   
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=10)   
# 拆分文档   
documents = text_splitter.split_documents(documents)   

from langchain.embeddings import HuggingFaceBgeEmbeddings   
from langchain.vectorstores import Chroma      
# embedding model: m3e-base   
model_name = "moka-ai/m3e-base"   
model_kwargs = {'device': 'cuda'}   
encode_kwargs = {'normalize_embeddings': True}   
embedding = HuggingFaceBgeEmbeddings(       
                                     model_name=model_name,       
                                     model_kwargs=model_kwargs,       
                                     encode_kwargs=encode_kwargs   
                                     )   

# 指定 persist_directory 将会把嵌入存储到磁盘上。   
persist_directory = 'db'   
db = Chroma.from_documents(documents, embedding, persist_directory=persist_directory)   

retriever = db.as_retriever()   

from langchain.prompts import ChatPromptTemplate      
template = """You are an assistant for question-answering tasks.    Use the following pieces of retrieved context to answer the question.    If you don't know the answer, just say that you don't know.    Use three sentences maximum and keep the answer concise.   Question: {question}    Context: {context}    Answer:   """   
prompt = ChatPromptTemplate.from_template(template)   

from langchain.schema.runnable import RunnablePassthrough   
from langchain.schema.output_parser import StrOutputParser     
from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key='sk-e39450eb1e1d4a8e825df0a7e4f5f411',
    openai_api_base='https://api.deepseek.com',
    max_tokens=100
)

# Input a single prompt
# prompt = "Once upon a time"
 
# Generate text based on the prompt
# generated_text = chat_model.invoke(prompt)
# print(generated_text)

# llm = ChatOllama(model='llama3')
rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | chat_model | StrOutputParser())    
query = "What is the Sutton's opinion on AI research at that time?"   
# print(retriever.get_relevant_documents("What is the Sutton's opinion on AI research at that time?"))
response = rag_chain.invoke(query)   
print(response)   

# query = "What is the Sutton's opinion on AI research at that time?"  
# Sutton's opinion on AI research at that time is that general methods leveraging computation are ultimately the most effective. He argues that while researchers often focus on leveraging human knowledge for short-term gains, the long-term success of AI depends on the ability to harness increasing computational power.