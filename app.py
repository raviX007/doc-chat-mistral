import os
import together
import streamlit as st
from typing import Any, Dict, Optional, List
from langchain.llms.base import LLM
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import Field
import tempfile
from langchain_core.prompts import PromptTemplate

class TogetherLLM(LLM):
    model_name: str = Field(default="togethercomputer/Llama-2-7B-32K-Instruct")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=512)
    together_api_key: str = Field()

    def __init__(
        self,
        model_name: str = "togethercomputer/Llama-2-7B-32K-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
    ):
        if not api_key:
            raise ValueError("API key is required")
        
        super().__init__(
            together_api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        together.api_key = self.together_api_key

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        response = together.Complete.create(
            prompt=prompt,
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=stop,
        )
        
        try:
            # First try the new API response format
            return response['choices'][0]['text']
        except (KeyError, TypeError):
            try:
                # Try the old/alternative format
                return response['output']['choices'][0]['text']
            except (KeyError, TypeError):
                # If both fail, return the raw response as string
                return str(response)

    @property
    def _llm_type(self) -> str:
        return "together"

def setup_database(uploaded_files):
    if not uploaded_files:
        st.error("Please upload PDF files")
        return None

    documents = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(temp_path)
            documents.extend(loader.load())

    texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
   
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en",
        model_kwargs={'device': 'cpu'}
    )

    os.makedirs('db', exist_ok=True)
    return Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory='db')

def main():
    st.set_page_config(page_title="RAG with Mistral", layout="wide")
   
    st.title("📚 RAG with Mistral")

    # Add API key input field in sidebar
    with st.sidebar:
        api_key = st.text_input("Enter Together AI API Key:", type="password")
        if not api_key:
            st.warning("Please enter your Together AI API key to continue")
            return
    
    # Move the rest of the UI below API key check
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        accept_multiple_files=True,
        type=['pdf']
    )
   
    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None
       
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    if uploaded_files:
        with st.spinner("Processing documents..."):
            vectordb = setup_database(uploaded_files)
            if vectordb:
                st.session_state.vectordb = vectordb
                retriever = vectordb.as_retriever(search_kwargs={"k": 5})

                llm = TogetherLLM(
                    model_name="mistralai/Mistral-7B-Instruct-v0.2",
                    temperature=0.1,
                    max_tokens=1024,
                    api_key=api_key  # Pass the API key from UI
                )

                prompt = PromptTemplate(
                    template="<s>[INST] Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {question} [/INST]",
                    input_variables=["context", "question"]
                )

                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt},
                    return_source_documents=True
                )
                st.success("Documents processed successfully!")

    if st.session_state.qa_chain:
        query = st.text_input("Ask a question about your documents:")
        if query:
            with st.spinner("Generating answer..."):
                response = st.session_state.qa_chain(query)
               
                st.subheader("Answer:")
                st.write(response['result'])
               
                st.subheader("Sources:")
                for source in response["source_documents"]:
                    with st.expander(f"Source: {source.metadata['source']}"):
                        st.write(source.page_content)

    else:
        st.info("Upload PDF documents to get started!")

if __name__ == "__main__":
    main()