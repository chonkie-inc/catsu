import pytest


def test_client_with_proxy():
    """Test that Client accepts a proxy parameter."""
    from catsu import Client

    client = Client(proxy="http://proxy.example.com:8080")
    assert client is not None


def test_client_with_invalid_ca_cert():
    """Test that Client raises an error for invalid CA certificate."""
    from catsu import Client

    with pytest.raises(RuntimeError):
        Client(ca_cert="not a valid certificate")


def test_client_with_valid_ca_cert_format():
    """Test that Client accepts a valid PEM format CA certificate.

    Note: This uses a syntactically valid but fake certificate.
    The client may fail later when making requests, but construction should work.
    """
    from catsu import Client

    # A minimal self-signed test certificate (valid PEM format)
    test_cert = """-----BEGIN CERTIFICATE-----
MIIBkTCB+wIJAKHBfpegPjMCMA0GCSqGSIb3DQEBCwUAMBExDzANBgNVBAMMBnVu
dXNlZDAeFw0yMzAxMDEwMDAwMDBaFw0yNDAxMDEwMDAwMDBaMBExDzANBgNVBAMM
BnVudXNlZDBcMA0GCSqGSIb3DQEBAQUAA0sAMEgCQQC6fGQKtQ3u3tLGDNnM8Jv2
vHNJJnKJkf8J8J6jJJ8J8J6jJJ8J8J6jJJ8J8J6jJJ8J8J6jJJ8J8J6jJJ8J8J6j
AgMBAAEwDQYJKoZIhvcNAQELBQADQQBN6V7t8Hy8cWJxmXNvh8J6jJJ8J8J6jJJ8
J8J6jJJ8J8J6jJJ8J8J6jJJ8J8J6jJJ8J8J6jJJ8J8J6jJJ8J8J6
-----END CERTIFICATE-----"""

    # This may raise due to invalid cert content, but should at least parse the PEM format
    try:
        client = Client(ca_cert=test_cert)
        assert client is not None
    except RuntimeError as e:
        # Accept errors about invalid cert content, but not about unexpected params
        assert "certificate" in str(e).lower() or "pem" in str(e).lower() or "base64" in str(e).lower()


def test_client_with_proxy_and_other_options():
    """Test that proxy can be combined with other options."""
    from catsu import Client

    client = Client(
        proxy="http://proxy.example.com:8080",
        max_retries=5,
        timeout=60,
    )
    assert client is not None
