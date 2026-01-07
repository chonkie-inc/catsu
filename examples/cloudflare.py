"""Example usage of catsu with Cloudflare Workers AI embeddings."""

from catsu import Client


def main():
    client = Client()

    if not client.has_provider("cloudflare"):
        print("Cloudflare provider not available.")
        print("Set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID environment variables.")
        return

    # Generate embeddings
    response = client.embed(
        "cloudflare:@cf/baai/bge-base-en-v1.5",
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
