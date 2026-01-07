"""Test to verify latency measurement and connection reuse."""

from catsu import CatsuClient


def main():
    client = CatsuClient()
    texts = ["Hello, world!"]

    print("Running 5 sequential requests to check connection reuse:\n")

    for i in range(5):
        response = client.embed("text-embedding-3-small", texts)
        print(f"Request {i + 1}: {response.latency_ms:.2f} ms")

    print("\n(First request is typically slower due to TLS handshake)")


if __name__ == "__main__":
    main()
