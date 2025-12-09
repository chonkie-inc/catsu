"""
VoyageAI Usage Examples

This example demonstrates how to use Mimie with VoyageAI embedding models.

Requirements:
    - Set VOYAGE_API_KEY environment variable
    - pip install mimie
"""

import asyncio
import os

from mimie import Client


def basic_usage():
    """Basic synchronous embedding example."""
    print("\n=== Basic Usage ===")

    # Initialize client
    client = Client()

    # Generate embeddings (auto-detects VoyageAI for voyage-3)
    response = client.embed(
        model="voyage-3",
        input="Hello, world!",
        input_type="query"
    )

    print(f"Model: {response.model}")
    print(f"Provider: {response.provider}")
    print(f"Dimensions: {response.dimensions}")
    print(f"Tokens used: {response.usage.tokens}")
    print(f"Cost: ${response.usage.cost:.6f}")
    print(f"Latency: {response.latency_ms:.2f}ms")
    print(f"Embedding (first 5 dims): {response.embeddings[0][:5]}")


def batch_embeddings():
    """Batch embedding example."""
    print("\n=== Batch Embeddings ===")

    client = Client()

    texts = [
        "Machine learning is fascinating",
        "Deep learning powers modern AI",
        "Natural language processing is evolving"
    ]

    response = client.embed(
        model="voyage-3-lite",
        input=texts,
        input_type="document"
    )

    print(f"Embedded {response.input_count} texts")
    print(f"Total tokens: {response.usage.tokens}")
    print(f"Total cost: ${response.usage.cost:.6f}")

    for i, embedding in enumerate(response.embeddings):
        print(f"Text {i+1}: {len(embedding)} dimensions")


def provider_specification():
    """Three ways to specify the provider."""
    print("\n=== Provider Specification Methods ===")

    client = Client()
    text = "Example text"

    # Method 1: Explicit provider parameter
    print("\n1. Explicit provider parameter:")
    response = client.embed(provider="voyageai", model="voyage-3", input=text)
    print(f"   ✓ {response.provider}:{response.model}")

    # Method 2: Provider prefix in model string
    print("\n2. Provider prefix:")
    response = client.embed(model="voyageai:voyage-3", input=text)
    print(f"   ✓ {response.provider}:{response.model}")

    # Method 3: Auto-detection (model name is unique)
    print("\n3. Auto-detection:")
    response = client.embed(model="voyage-3", input=text)
    print(f"   ✓ Auto-detected {response.provider}:{response.model}")


async def async_usage():
    """Async embedding example."""
    print("\n=== Async Usage ===")

    client = Client()

    # Single async embedding
    response = await client.aembed(
        model="voyage-3",
        input="Async embedding example"
    )

    print(f"Async embedding completed in {response.latency_ms:.2f}ms")

    # Parallel async embeddings
    tasks = [
        client.aembed(model="voyage-3", input=f"Text {i}")
        for i in range(3)
    ]

    responses = await asyncio.gather(*tasks)
    print(f"Processed {len(responses)} embeddings in parallel")

    await client.aclose()


def tokenization():
    """Token counting without embedding."""
    print("\n=== Tokenization ===")

    client = Client()

    texts = [
        "Short text",
        "This is a longer text with more tokens to count"
    ]

    for text in texts:
        # Count tokens using local tokenizer (no API call!)
        token_response = client._providers["voyageai"].tokenize(
            model="voyage-3",
            inputs=[text]
        )
        print(f"'{text[:30]}...' → {token_response.token_count} tokens")


def different_models():
    """Compare different VoyageAI models."""
    print("\n=== Different Models ===")

    client = Client()
    text = "Compare embedding models"

    models = ["voyage-3", "voyage-3-lite", "voyage-code-3"]

    for model in models:
        response = client.embed(model=model, input=text)
        print(f"{model:20s} - {response.dimensions} dims, "
              f"{response.usage.tokens} tokens, ${response.usage.cost:.6f}")


def list_available_models():
    """List all available VoyageAI models."""
    print("\n=== Available VoyageAI Models ===")

    client = Client()
    models = client.list_models(provider="voyageai")

    print(f"Found {len(models)} VoyageAI models:\n")
    for model in models:
        print(f"  {model['name']:25s} - {model['dimensions']} dims, "
              f"${model['cost_per_million_tokens']}/M tokens")


def error_handling():
    """Error handling examples."""
    print("\n=== Error Handling ===")

    from mimie.utils.errors import ModelNotFoundError, InvalidInputError

    client = Client()

    # Handle invalid model
    try:
        client.embed(model="nonexistent-model", input="test")
    except ModelNotFoundError as e:
        print(f"✓ Caught ModelNotFoundError: {e.model}")

    # Handle invalid input
    try:
        client.embed(model="voyage-3", input="")
    except InvalidInputError as e:
        print(f"✓ Caught InvalidInputError: {e.parameter}")


def main():
    """Run all examples."""
    # Check API key
    if not os.getenv("VOYAGE_API_KEY"):
        print("⚠️  Please set VOYAGE_API_KEY environment variable")
        print("   export VOYAGE_API_KEY='your-api-key-here'")
        return

    print("🌐 Mimie + VoyageAI Examples 🚀")
    print("=" * 50)

    # Run synchronous examples
    basic_usage()
    batch_embeddings()
    provider_specification()
    different_models()
    list_available_models()
    tokenization()
    error_handling()

    # Run async example
    print("\nRunning async examples...")
    asyncio.run(async_usage())

    print("\n" + "=" * 50)
    print("✅ All examples completed!")


if __name__ == "__main__":
    main()
