"""Example demonstrating query vs document embeddings with catsu."""

from catsu import CatsuClient


def main():
    client = CatsuClient()

    # Embed a query (for retrieval)
    query_response = client.embed(
        "text-embedding-3-small",
        ["What is the capital of France?"],
        input_type="query",
    )
    print(f"Query embedding - Input type: {query_response.input_type}")
    print(f"Latency: {query_response.latency_ms:.2f} ms\n")

    # Embed documents (for indexing)
    doc_response = client.embed(
        "text-embedding-3-small",
        [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
            "Tokyo is the capital of Japan.",
        ],
        input_type="document",
    )
    print(f"Document embeddings - Input type: {doc_response.input_type}")
    print(f"Documents embedded: {doc_response.input_count}")
    print(f"Latency: {doc_response.latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
