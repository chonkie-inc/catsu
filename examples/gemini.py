"""Example usage of catsu with Google Gemini embeddings."""

from catsu import Client


def main():
    client = Client()

    if not client.has_provider("gemini"):
        print("Gemini provider not available. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        return

    # Generate embeddings
    response = client.embed(
        "gemini:text-embedding-004",
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
