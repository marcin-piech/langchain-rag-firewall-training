import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_firecrawl")


def create_vector_store():
    """Crawl the website, split the content, create embeddings, and persist the vector store."""
    # Define the Firecrawl API key
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY environment variable not set")

    # Step 1: Crawl the website using FireCrawlLoader
    print("Begin crawling the website...")
    loader = FireCrawlLoader(
        api_key=api_key, url="https://booksy.com/pl-pl/s/fryzjer/20383_gdansk", mode="scrape")
    docs = loader.load()
    print("Finished crawling the website.")

    # Convert metadata values to strings if they are lists
    for doc in docs:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))

    # Step 2: Split the crawled content into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(split_docs)}")
    print(f"Sample chunk:\n{split_docs[0].page_content}\n")

    # Step 3: Create embeddings for the document chunks
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Step 4: Create and persist the vector store with the embeddings
    print(f"\n--- Creating vector store in {persistent_directory} ---")
    db = Chroma.from_documents(
        split_docs, embeddings, persist_directory=persistent_directory
    )
    print(f"--- Finished creating vector store in {persistent_directory} ---")


# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    create_vector_store()
else:
    print(
        f"Vector store {persistent_directory} already exists. No need to initialize.")

# Load the vector store with the embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)


# Step 5: Query the vector store
# def query_vector_store(query):
#     """Query the vector store with the specified question."""
#     # Create a retriever for querying the vector store
#     retriever = db.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 3},
#     )

#     # Retrieve relevant documents based on the query
#     relevant_docs = retriever.invoke(query)

#     # Display the relevant results with metadata
#     print("\n--- Relevant Documents ---")
#     for i, doc in enumerate(relevant_docs, 1):
#         print(f"Document {i}:\n{doc.page_content}\n")
#         if doc.metadata:
#             print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


def ask_llm_with_context(question):
    """Use GPT model to answer the question using retrieved documents as context."""
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 15},
    )

    # Define the LLM (GPT model)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Create QA chain with retriever and LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Ask the question
    result = qa_chain.invoke({"query": question})

    # Print the result
    print("\n--- Answer from GPT ---")
    print(result["result"])

    # Optionally print source documents
    print("\n--- Sources ---")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"Source {i}: {doc.metadata.get('source', 'Unknown')}")


# Define the user's question
# system_message = "You are a chatbot dedicated to Amazon website with backpacks. If a user asks you about anything not related with the backpacks please answer 'I cannot answer the message:( '. "

# human_message1 = "Podaj 3 linki do usług ocenianych conajmniej na 4.7 przez klientów."

human_message = "Ile kosztuje czerwony rower."

ask_llm_with_context(human_message)
