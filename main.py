pinecone_api_key =  "ebfd4dc6-ce8b-4516-a1b0-220fd6700bf7"
pinecone_index = "pdfchat"
pinecone_namespace = "pdf_ask"
api_key = "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"

import os 

os.environ["PINECONE_API_KEY"] = pinecone_api_key
from PyPDF2 import PdfReader
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS

memory = ConversationBufferWindowMemory(max_size=5)
embedder = OpenAIEmbeddings(openai_api_key=api_key,model="text-embedding-3-large",dimensions=3072)
model = ChatOpenAI(openai_api_key=api_key,model="gpt-3.5-turbo")
splitter = RecursiveCharacterTextSplitter(chunk_size=3000,chunk_overlap=400)


def get_chunks_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    splitter = RecursiveCharacterTextSplitter()
    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        chunks.extend(splitter.split_text(text))
    return chunks


def get_vector_store(uploaded_pdf):
    texts = get_chunks_from_pdf(uploaded_pdf)
    metadata = [{"file_name": uploaded_pdf.name,"chunk_id":idx} for idx in range(len(texts))]
    store = FAISS.from_texts(texts, embedding=embedder, metadatas=metadata)
    return store

def get_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, don't provide the wrong answer. Although if the user ask about their previous question return their chat history.\n\n
    You get the user's chat history along with the context and question, if the user asks about a previous question, you can refer to the chat history\n\n
    Users's chat history: \n{chat_history}\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question","chat_history"])
    chain = prompt |  model
    return chain

def get_answer(prompt,store):
    chain = get_chain()
    docs = store.similarity_search(prompt,k=5)
    context = " ".join([doc.page_content for doc in docs])
    
    answer = chain.invoke({"context":context,
                        "question":prompt,
                        "chat_history":memory.load_memory_variables({})}).content
    memory.save_context({"input":prompt},{"output":answer}) 
    print(memory.load_memory_variables({}))
    print(context)
    return answer


def extract_data(file):
    pdf_reader = PdfReader(file)
    # Extract the content
    content = ""
    for page in range(len(pdf_reader.pages)):
        content += pdf_reader.pages[page].extract_text()
    return content


def main():
    st.title("Ask your pdf")
   

    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
    if uploaded_file is not None:
        vector_store = get_vector_store(uploaded_file)
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("What is up?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            response = get_answer(prompt,vector_store)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

   
if __name__ == "__main__":
    main()