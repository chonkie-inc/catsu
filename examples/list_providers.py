"""List available providers based on configured API keys."""

from catsu import Client


def main():
    client = Client()

    providers = client.list_providers()
    print(f"Available providers ({len(providers)}):")
    for provider in sorted(providers):
        print(f"  - {provider}")

    print("\nSupported providers (set API key to enable):")
    all_providers = [
        ("openai", "OPENAI_API_KEY"),
        ("voyageai", "VOYAGE_API_KEY"),
        ("cohere", "COHERE_API_KEY"),
        ("jina", "JINA_API_KEY"),
        ("mistral", "MISTRAL_API_KEY"),
        ("gemini", "GOOGLE_API_KEY or GEMINI_API_KEY"),
        ("together", "TOGETHER_API_KEY"),
        ("mixedbread", "MIXEDBREAD_API_KEY"),
        ("nomic", "NOMIC_API_KEY"),
        ("deepinfra", "DEEPINFRA_API_KEY"),
        ("cloudflare", "CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID"),
    ]
    for name, env_var in all_providers:
        status = "✓" if client.has_provider(name) else "✗"
        print(f"  [{status}] {name:12} ({env_var})")


if __name__ == "__main__":
    main()
