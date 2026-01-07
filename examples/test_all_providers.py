"""Test all enabled providers to verify they work correctly."""

from catsu import Client


# Model mappings for each provider
PROVIDER_MODELS = {
    "openai": "text-embedding-3-small",
    "voyageai": "voyage-3-lite",
    "cohere": "embed-english-v3.0",
    "jina": "jina-embeddings-v3",
    "mistral": "mistral-embed",
    # "gemini": "text-embedding-004",  # Requires special API key setup
    "together": "BAAI/bge-base-en-v1.5",
    "mixedbread": "mxbai-embed-large-v1",
    "nomic": "nomic-embed-text-v1.5",
    "deepinfra": "BAAI/bge-base-en-v1.5",
    "cloudflare": "@cf/baai/bge-base-en-v1.5",
}


def test_provider(client: Client, provider: str, model: str) -> dict:
    """Test a single provider and return results."""
    try:
        response = client.embed(
            f"{provider}:{model}",
            "Hello, world!",
        )
        return {
            "success": True,
            "dimensions": response.dimensions,
            "latency_ms": response.latency_ms,
            "tokens": response.usage.tokens,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def main():
    client = Client()
    providers = client.list_providers()

    print(f"Testing {len(providers)} enabled providers...\n")
    print("-" * 70)

    results = {}
    for provider in sorted(providers):
        model = PROVIDER_MODELS.get(provider)
        if not model:
            print(f"⚠️  {provider:12} - No model configured for testing")
            continue

        print(f"🔄 {provider:12} - Testing with {model}...", end=" ", flush=True)
        result = test_provider(client, provider, model)
        results[provider] = result

        if result["success"]:
            print(
                f"✓ {result['dimensions']}d, "
                f"{result['latency_ms']:.0f}ms, "
                f"{result['tokens']} tokens"
            )
        else:
            print(f"✗ {result['error'][:50]}...")

    print("-" * 70)

    # Summary
    successful = sum(1 for r in results.values() if r["success"])
    failed = sum(1 for r in results.values() if not r["success"])

    print(f"\nSummary: {successful} passed, {failed} failed out of {len(results)} tested")

    if failed > 0:
        print("\nFailed providers:")
        for provider, result in results.items():
            if not result["success"]:
                print(f"  - {provider}: {result['error']}")


if __name__ == "__main__":
    main()
