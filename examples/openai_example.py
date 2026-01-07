"""Simple example demonstrating catsu with OpenAI embeddings."""

from catsu import Client


def main():
    # Create client (reads OPENAI_API_KEY from environment)
    client = Client()

    # Generate embeddings
    response = client.embed(
        "text-embedding-3-small",
        ["Hello, world!", "How are you today?", "Rust is fast!"],
    )

    # Print response details
    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Dimensions: {response.dimensions}")
    print(f"Input count: {response.input_count}")
    print(f"Input type: {response.input_type}")
    print(f"Latency: {response.latency_ms:.2f} ms")
    print(f"Total tokens: {response.usage.tokens}")
    print(f"Cost: ${response.usage.cost:.6f}" if response.usage.cost else "Cost: N/A")
    print(f"\nEmbeddings shape: {len(response.embeddings)} x {len(response.embeddings[0])}")

    # Show first few values of first embedding
    print(f"First embedding (first 5 values): {response.embeddings[0][:5]}")


if __name__ == "__main__":
    main()
