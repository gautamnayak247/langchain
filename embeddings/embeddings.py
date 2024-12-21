from langchain import OpenAI
#vector store
from langchain.vectorstores import FAISS

#lanchain component we will be using to get the documents
from langchain.chains import RetrievalQA

#document loader for text
from langchain.document_loaders import TextLoader

#embeddings
from langchain.embeddings import OpenAIEmbeddings

# text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter




loader = TextLoader("embeddings\data\worked.txt")
doc = loader.load()

#splitting to chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = text_splitter.split_documents(doc)


embeddings = OpenAIEmbeddings(open_ai_key="")

docsearch = FAISS.from_documents(docs, embeddings)

#create retrieval engine
llm = OpenAI(temperature=0, openai_api_key="")
qa = RetrievalQA.from_chain_type(llm = llm, chain_type="stuff", retriever = docsearch.as_retriever())   

query = "What does the author describe as good work?"
qa.run(query)
