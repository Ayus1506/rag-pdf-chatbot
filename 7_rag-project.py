from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

if "document_upload" not in st.session_state:
    st.session_state.document_upload=False

if "agent" not in st.session_state:
    st.session_state.agent=None

if "vector_store" not in st.session_state:
    st.session_state.vector_store=None

if "messages" not in st.session_state:
    st.session_state.messages=[]

def process_document(path):
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    splitters = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=200)
    splitted_data=splitters.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    vector_db = InMemoryVectorStore.from_documents(
        documents=splitted_data,
        embedding=embeddings
    )


    llm=ChatGroq(model="openai/gpt-oss-120b")

    @tool
    def retriever(query:str):
        """ 
        this tool can help you to retrieve relevant data of pdf document 
        """
        docs=vector_db.similarity_search(query=query)
        context = ""
        for doc in docs:
            context +=doc.page_content + '\n'

        return context

    system_prompt = """
    you are a helpful assistant that answer question using retriever context
    always use the `retriever` tool for questions requiring external knowledge 
    """
    memory=InMemorySaver()
    agent = create_agent(
        model=llm,
        tools=[retriever],
        system_prompt=system_prompt,
        checkpointer=memory

    )
    st.session_state.agent=agent
    st.session_state.document_upload=True


if st.session_state.document_upload == False:
    uploaded=st.file_uploader(label='upload your file' , type=['pdf'] , accept_multiple_files=True)
    if uploaded:
        with st.spinner('processing...'):
           path="./doc_files/" 
           for file in uploaded:
               with open(path + file.name, "wb") as f:
                   f.write(file.getvalue())
            
            
           process_document(path)
           st.rerun()


if st.session_state.document_upload and st.session_state.agent:

    for message in st.session_state.messages:
        role=message.get('role')
        content=message.get('content')
        st.chat_message(role).markdown(content)




    query=st.chat_input('ask anything related to uploaded document...')
   
    if query:
        st.session_state.messages.append({'role':'user','content':query})
        st.chat_message('user').markdown(query)
        response=st.session_state.agent.invoke(
            {'messages':[{'role':'user','content':query}]},
            {'configurable':{'thread_id':1}}
        )
        answer=response['messages'][-1].content
        st.chat_message('ai').markdown(answer)
        st.session_state.messages.append({'role':'ai','content':answer})
