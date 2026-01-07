"""Example usage of catsu with Together AI embeddings."""

from catsu import Client


def main():
    client = Client()

    if not client.has_provider("together"):
        print("Together provider not available. Set TOGETHER_API_KEY environment variable.")
        return

    # Generate embeddings
    response = client.embed(
        "together:BAAI/bge-base-en-v1.5",
        "Hello, world!",
    )

    print(f"Model: {response.model}")
    print(f"Provider: {response.provider}")
    print(f"Dimensions: {response.dimensions}")
    print(f"Input count: {response.input_count}")
    print(f"Latency: {response.latency_ms:.2f}ms")
    print(f"Total tokens: {response.usage.tokens}")
    print(f"First embedding (first 5 dims): {response.embeddings[0][:5]}")


if __name__ == "__main__":
    main()
