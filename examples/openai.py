"""OpenAI Usage Examples.

This example demonstrates how to use Catsu with OpenAI embedding models.

Requirements:
    - Set OPENAI_API_KEY environment variable
    - pip install catsu
"""

import asyncio
import os

from catsu import Client


def basic_usage():
    """Give basic synchronous embedding example."""
    print("\n=== Basic Usage ===")

    # Initialize client
    client = Client()

    # Generate embeddings (auto-detects OpenAI for text-embedding-3-small)
    response = client.embed(model="text-embedding-3-small", input="Hello, world!")

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
        "Natural language processing is evolving",
    ]

    response = client.embed(model="text-embedding-3-small", input=texts)

    print(f"Embedded {response.input_count} texts")
    print(f"Total tokens: {response.usage.tokens}")
    print(f"Total cost: ${response.usage.cost:.6f}")

    for i, embedding in enumerate(response.embeddings):
        print(f"Text {i + 1}: {len(embedding)} dimensions")


def custom_dimensions():
    """Use custom output dimensions (text-embedding-3 models only)."""
    print("\n=== Custom Dimensions ===")

    client = Client()
    text = "Reduce embedding dimensions for efficiency"

    # Default dimensions for text-embedding-3-large is 3072
    response_full = client.embed(model="text-embedding-3-large", input=text)
    print(f"Full dimensions: {response_full.dimensions}")

    # Reduce to 256 dimensions
    response_256 = client.embed(model="text-embedding-3-large", input=text, dimensions=256)
    print(f"Reduced to 256: {response_256.dimensions}")

    # Reduce to 1024 dimensions
    response_1024 = client.embed(
        model="text-embedding-3-large", input=text, dimensions=1024
    )
    print(f"Reduced to 1024: {response_1024.dimensions}")


def provider_specification():
    """Three ways to specify the provider."""
    print("\n=== Provider Specification Methods ===")

    client = Client()
    text = "Example text"

    # Method 1: Explicit provider parameter
    print("\n1. Explicit provider parameter:")
    response = client.embed(provider="openai", model="text-embedding-3-small", input=text)
    print(f"   Provider: {response.provider}, Model: {response.model}")

    # Method 2: Provider prefix in model string
    print("\n2. Provider prefix:")
    response = client.embed(model="openai:text-embedding-3-small", input=text)
    print(f"   Provider: {response.provider}, Model: {response.model}")

    # Method 3: Auto-detection (model name is unique)
    print("\n3. Auto-detection:")
    response = client.embed(model="text-embedding-3-small", input=text)
    print(f"   Auto-detected {response.provider}:{response.model}")


async def async_usage():
    """Async embedding example."""
    print("\n=== Async Usage ===")

    client = Client()

    # Single async embedding
    response = await client.aembed(
        model="text-embedding-3-small", input="Async embedding example"
    )

    print(f"Async embedding completed in {response.latency_ms:.2f}ms")

    # Parallel async embeddings
    tasks = [
        client.aembed(model="text-embedding-3-small", input=f"Text {i}")
        for i in range(3)
    ]

    responses = await asyncio.gather(*tasks)
    print(f"Processed {len(responses)} embeddings in parallel")

    await client.aclose()


def tokenization():
    """Token counting without embedding."""
    print("\n=== Tokenization ===")

    client = Client()

    texts = ["Short text", "This is a longer text with more tokens to count"]

    for text in texts:
        # Count tokens using local tiktoken tokenizer (no API call!)
        token_response = client._providers["openai"].tokenize(
            model="text-embedding-3-small", inputs=[text]
        )
        print(f"'{text[:30]}...' -> {token_response.token_count} tokens")


def different_models():
    """Compare different OpenAI embedding models."""
    print("\n=== Different Models ===")

    client = Client()
    text = "Compare embedding models"

    models = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]

    for model in models:
        response = client.embed(model=model, input=text)
        print(
            f"{model:25s} - {response.dimensions} dims, "
            f"{response.usage.tokens} tokens, ${response.usage.cost:.6f}"
        )


def list_available_models():
    """List all available OpenAI embedding models."""
    print("\n=== Available OpenAI Models ===")

    client = Client()
    models = client.list_models(provider="openai")

    print(f"Found {len(models)} OpenAI models:\n")
    for model in models:
        print(
            f"  {model['name']:25s} - {model['dimensions']} dims, "
            f"${model['cost_per_million_tokens']}/M tokens"
        )


def error_handling():
    """Error handling examples."""
    print("\n=== Error Handling ===")

    from catsu.utils.errors import InvalidInputError, ModelNotFoundError

    client = Client()

    # Handle invalid model
    try:
        client.embed(model="nonexistent-model", input="test")
    except ModelNotFoundError as e:
        print(f"Caught ModelNotFoundError: {e.model}")

    # Handle invalid input
    try:
        client.embed(model="text-embedding-3-small", input="")
    except InvalidInputError as e:
        print(f"Caught InvalidInputError: {e.parameter}")

    # Handle invalid dimensions parameter
    try:
        # ada-002 doesn't support custom dimensions
        client.embed(model="text-embedding-ada-002", input="test", dimensions=256)
    except InvalidInputError as e:
        print(f"Caught InvalidInputError: {e.message}")


def main():
    """Run all examples."""
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return

    print("Catsu + OpenAI Examples")
    print("=" * 50)

    # Run synchronous examples
    basic_usage()
    batch_embeddings()
    custom_dimensions()
    provider_specification()
    different_models()
    list_available_models()
    tokenization()
    error_handling()

    # Run async example
    print("\nRunning async examples...")
    asyncio.run(async_usage())

    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    main()
