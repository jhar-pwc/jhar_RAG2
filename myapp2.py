## RAG Q&A Conversation With PDF Including Chat History

import streamlit as st
from langchain.chains.history_aware_retriever import   create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
#from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
#from gen_ai_hub.proxy.native.openai import embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
#from gen_ai_hub.proxy.langchain.init_models import  init_embedding_model
import chromadb
import os
#from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from dotenv import load_dotenv
load_dotenv()

#os.environ['OPENAI_API_BASE'] = "https://api.ai.prod.us-east-1.aws.ml.hana.ondemand.com/v2/inference/deployments/openai-embedding/v1"

 
###

#proxy_client = get_proxy_client('gen-ai-hub')
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
#st.secrets("GROQ_API_KEY")
#st.secrets("OPENAI_API_KEY")
# Create Open AI embeddingsvsing Gen AI HUB
#embeddings = init_embedding_model('text-embedding-ada-002')
embeddings = OpenAIEmbeddings()
#embeddings = OllamaEmbeddings()


st.set_page_config(page_title=" A Conversational RAG  Application ", page_icon="🧊",

    layout="wide",
    initial_sidebar_state="expanded",

    )

st.header(" A GenAI Assistant for INtegration  Assessment Projects!")
st.title("🦜🔗 A Conversational RAG  Application using PwC internal Knowledge base")
Select_llm = st.sidebar.selectbox("Select an OpenAI model", ["gpt-4o","gpt-4-turbo","gpt-4","gpt-3.5-turbo"])
st.write("Go ahead and ask me any Question related to Integration Assessment")
temperature = st.sidebar.slider("Temperature", min_value=0.0,max_value=1.0,value=0.7 )
max_tokens = st.sidebar.slider("Max_Tokens", min_value=0.0,max_value=1.0,value=0.7 )
my_middleware = st.multiselect('Choose the Integration migration patterns ', ['GreenField','BrownField',"BlueField"])

## Input the Groq API Key

api_key=st.text_input("Enter your API key:",type="password")

 

## Check if groq api key is provided

if api_key:

    llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.1-8b-instant")

    #llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, max_tokens=None, timeout=None, max_retries=2,)

    ## chat interface
    session_id=st.text_input("Sesion ID",value="default_session")
    ## statefully manage chat history 

    if 'store' not in st.session_state:
        st.session_state.store={} 

    uploaded_files=st.file_uploader("Choose the PwC internal  knowledgebase Doc for Model Grounding",type="pdf",accept_multiple_files=True)

    ## Process uploaded  PDF's
    if uploaded_files:

        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"

            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

 

    # Split and create embeddings for the documents

        chromadb.api.client.SharedSystemClient.clear_system_cache()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)

        splits = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

        #vectorstore1 =FAISS.from_documents(documents,embeddings)

        retriever = vectorstore.as_retriever()    

 

        contextualize_q_system_prompt=(

            "Given a chat history and the latest user question"

            "which might reference context in the chat history, "

            "formulate a standalone question which can be understood "

            "without the chat history. Do NOT answer the question, "

            "just reformulate it if needed and otherwise return it as is."

        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(

                [

                    ("system", contextualize_q_system_prompt),

                    MessagesPlaceholder("chat_history"),

                    ("human", "{input}"),

                ]

            )

       

        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

 

            # Answer question

        system_prompt = (

                "You are an assistant for question-answering tasks. "

                "Use the following pieces of retrieved context to answer "

                "the question. If you don't know the answer, say that you "

                "don't know. Use six  sentences maximum and keep the "

                "answer concise. If you don't know the answer, answer with Unfortunately, I don't have the information."

                "\n\n"

                "{context}"

            )

        qa_prompt = ChatPromptTemplate.from_messages(

                [

                    ("system", system_prompt),

                    MessagesPlaceholder("chat_history"),

                    ("human", "{input}"),

                ]

            )

       

        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)

        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

 

        def get_session_history(session:str)->BaseChatMessageHistory:

            if session_id not in st.session_state.store:

                st.session_state.store[session_id]=ChatMessageHistory()

            return st.session_state.store[session_id]

       

        conversational_rag_chain=RunnableWithMessageHistory(

            rag_chain,get_session_history,

            input_messages_key="input",

            history_messages_key="chat_history",

            output_messages_key="answer"

        )

 

        user_input = st.text_input("Ask me any question:")

        if user_input:

            session_history=get_session_history(session_id)

            response = conversational_rag_chain.invoke(

                {"input": user_input},

                config={

                    "configurable": {"session_id":session_id}

                },  # constructs a key "abc123" in `store`.

            )

            #st.write(st.session_state.store)

            #st.write(response['answer'])

            st.write("Assistant:", response['answer'])

            st.write("Chat History:", session_history.messages)

else:

    st.warning("Please enter Your Authentication parameter (OPen API Key) to proceed",icon="⚠")

 


