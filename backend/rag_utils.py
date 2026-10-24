
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant answering questions about hotel policies using
    only the information provided in the retrieved documentation.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only the explicit information provided in the 'Retrieved Policy Information'. Do not use outside knowledge or make assumptions.
    2. If the retrieved context does not contain enough details to confidently and completely answer the query, state: 'The available policy documents do not contain enough information to answer this question.'
    3. Always search for property-specific rules, exceptions, and conditions (such as specific fees, pet rules, or check-in times for a given property) and present them clearly.
    4. If the retrieved documentation contains conflicting information, clearly identify the conflict rather than choosing an answer based on outside knowledge.
    5. Keep the response accurate, concise, factual, and aligned strictly with the provided text.
    6. Do not invent any rules, figures, numbers, or terms.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 5,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
