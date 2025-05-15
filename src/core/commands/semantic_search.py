from typing import List
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import chain


def run(model):
    documents = [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"source": "mammal-pets-doc"},
        ),
    ]

    file_path = "data/nke-10k-2023.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    print(len(docs))

    print(f"{docs[0].page_content[:200]}\n")
    print(docs[0].metadata)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    print(len(all_splits))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_1 = embeddings.embed_query(all_splits[0].page_content)
    vector_2 = embeddings.embed_query(all_splits[1].page_content)

    assert len(vector_1) == len(vector_2)
    print(f"Generated vectors of length {len(vector_1)}\n")
    print(vector_1[:10])

    vector_store = InMemoryVectorStore(embeddings)

    ids = vector_store.add_documents(documents=all_splits)

    # Synchronous search
    results = vector_store.similarity_search(
        "How many distribution centers does Nike have in the US?"
    )

    print(results[0])

    # Asynchronous search
    # results = await vector_store.asimilarity_search("When was Nike incorporated?")

    # print(results[0])

    # Note that providers implement different scores; the score here
    # is a distance metric that varies inversely with similarity.

    results = vector_store.similarity_search_with_score(
        "What was Nike's revenue in 2023?"
    )
    doc, score = results[0]
    print(f"Score: {score}\n")
    print(doc)

    embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")

    results = vector_store.similarity_search_by_vector(embedding)
    print(results[0])

    @chain
    def retriever(query: str) -> List[Document]:
        return vector_store.similarity_search(query, k=1)

    documents_retrieved = retriever.batch(
        [
            "How many distribution centers does Nike have in the US?",
            "When was Nike incorporated?",
        ],
    )

    print(f"Retrieved {len(documents_retrieved)} documents\n")
    for doc in documents_retrieved:
        print(doc[0].page_content[:200])

    # or equivalently:
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    documents_retrieved = retriever.batch(
        [
            "How many distribution centers does Nike have in the US?",
            "When was Nike incorporated?",
        ],
    )

    print(f"Retrieved {len(documents_retrieved)} documents\n")
    for doc in documents_retrieved:
        print(doc[0].page_content[:200])

    pass
