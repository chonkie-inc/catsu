# Why Embedding API Clients Are Broken

This document elaborates on the problems with the current embedding API client ecosystem, providing concrete evidence for each claim made in the [README](../README.md).

## Table of Contents

- [OpenAI's Client Wasn't Designed for Embeddings](#openais-client-wasnt-designed-for-embeddings)
- [Provider-Specific Libraries Are Inconsistent or Broken](#provider-specific-libraries-are-inconsistent-or-broken)
- [Universal Clients Don't Focus on Embeddings](#universal-clients-dont-focus-on-embeddings)
- [Capability Inconsistencies Across Providers](#capability-inconsistencies-across-providers)
- [Missing Basic Features](#missing-basic-features)
- [No Single Source of Truth for Model Metadata](#no-single-source-of-truth-for-model-metadata)

---

## OpenAI's Client Wasn't Designed for Embeddings

Everyone defaults to OpenAI's client for embeddings, but it has fundamental limitations for embedding-specific workloads.

### Token Limit Handling Is Broken

LangChain's `OpenAIEmbeddings` class [doesn't respect token limits](https://github.com/langchain-ai/langchain/issues/31227), causing 400 BadRequest errors. The expected behavior is that the class should internally batch requests to stay under OpenAI's max tokens per request limit (currently 300,000 tokens), but it doesn't.

### Undocumented Per-Request Limits

Users have discovered a ~300K token per-request limit (approximately 36 inputs × 8K tokens each), but there's [no official documentation](https://community.openai.com/t/max-total-embeddings-tokens-per-request/1254699) for this limit. Developers are left guessing whether this varies by model or might change in the future.

### Rate Limit Inconsistencies

Users report [429 rate limit errors](https://community.openai.com/t/rate-limit-reached-with-large-documents/358525) even when processing documents well under the stated limits. The token-per-minute limits don't apply consistently—sometimes errors resolve without waiting the stated cooldown period.

[Azure OpenAI users](https://learn.microsoft.com/en-us/answers/questions/1693832/azure-openai-error-429-request-below-rate-limit) encounter the same issues, receiving `openai.RateLimitError: Error code: 429` on first calls, before any rate could reasonably be exceeded.

### Reliability Degradation

Long-time users of `text-embedding-ada-002` [started experiencing timeouts and bad requests](https://community.openai.com/t/issue-with-bad-requests-and-api-timeout-in-embedding-calls/954480) in September 2024 after 1.5+ years of stable usage, with no changes on their end.

---

## Provider-Specific Libraries Are Inconsistent or Broken

### VoyageAI SDK

**UnboundLocalError in Retry Logic**: The VoyageAI Python SDK had a bug causing `UnboundLocalError` in retry mechanisms, [fixed in v0.3.5](https://github.com/voyage-ai/voyageai-python/releases) (September 2024). This indicates insufficient testing of core functionality.

**Integration Failures**: Users attempting to use VoyageAI with Weaviate Cloud [encounter 422 errors](https://forum.weaviate.io/t/voyageai-text-embedding-in-weaviate-cloud-not-working/2543) (`UnexpectedStatusCodeError`) when creating collections with VoyageAI embeddings.

**Breaking Deprecation Changes**: The SDK shows warnings that the `model` argument defaults to a specific model but will become a required argument, breaking backward compatibility for existing code.

### Cohere SDK

**Breaking API Changes**: BERTopic v0.16 users with cohere v5.1.7 encounter a [TypeError in `Client.embed()`](https://github.com/MaartenGr/BERTopic/pull/1904): "takes 1 positional argument but 2 positional arguments (and 2 keyword-only arguments) were given." SDK updates changed the method signature without clear migration paths.

**Required Parameters Not Enforced**: The `input_type` parameter is required for embedding models v3 and higher, but [many integrations miss it](https://github.com/langchain-ai/langchain/issues/15234). LangChain's `MlflowEmbeddings` class was identified as lacking this required argument entirely.

**Incorrect Embeddings for Search**: Cohere prepends special tokens to differentiate embedding types (search_query vs search_document). Some implementations [don't correctly use the `input_type` parameter](https://github.com/lancedb/lancedb/issues/1329), leading to "noticeable performance reduction when evaluating search and retrieval pipelines."

**Azure Integration Broken**: When hosting Cohere embedding models on Azure AI, [the embedding client doesn't work](https://github.com/Azure/azure-sdk-for-python/issues/41050) because the endpoint returns an empty 200 response while the SDK expects actual embedding vectors.

---

## Universal Clients Don't Focus on Embeddings

### LiteLLM

LiteLLM is primarily designed for LLM completions, with embeddings as an afterthought.

**No Custom Provider Embeddings**: Developers building custom LLM providers [cannot implement embeddings](https://github.com/BerriAI/litellm/issues/6754). Attempts to use the embedding method within `BaseLLM` don't work as expected (issue opened November 2024).

**Proxy Embeddings Broken**: The `litellm_proxy` provider [isn't accounted for in embedding methods](https://github.com/BerriAI/litellm/issues/8077), causing errors. The workaround is passing `custom_llm_provider="openai"` instead of using the litellm_proxy prefix.

**Parameters Get Dropped**: LiteLLM [doesn't accept non-OpenAI parameters](https://github.com/BerriAI/litellm/issues/12110) for providers like DeepInfra. The only option is `drop_params`, which silently discards important configuration.

**Dimension Support Limited to OpenAI**: The `dimensions` parameter is [only supported for OpenAI/Azure text-embedding-3](https://docs.litellm.ai/docs/embedding/supported_embedding) models. Other providers' dimension parameters are not translated or supported.

### Mozilla's any-llm-sdk

Mozilla's any-llm provides a unified interface across providers but inherits the limitations of native SDKs.

**Relies on Native SDKs**: The library [leverages official SDKs when available](https://blog.mozilla.ai/introducing-any-llm-a-unified-api-to-access-any-llm-provider/), meaning it inherits all their bugs and inconsistencies documented above.

**Embeddings Added as Afterthought**: Embedding support was [added via feature request](https://github.com/mozilla-ai/any-llm/issues/33) (Issue #33) after initial release, suggesting it wasn't a core design consideration.

**API Inconsistency**: The [public API mimics LiteLLM](https://github.com/mozilla-ai/any-llm/issues/381), which itself has embedding limitations. The team acknowledged downsides including that "provider client gets re-initialized" for each call.

**Limited Provider Support**: Not all providers support embeddings through any-llm—users must check provider-specific documentation to see which ones do.

---

## Capability Inconsistencies Across Providers

Every provider has different capabilities with no standardized way to discover what's available.

### Dimension Parameter Chaos

| Provider | Parameter Name | Supported Models | Available Dimensions | Default |
|----------|---------------|------------------|---------------------|---------|
| **OpenAI** | `dimensions` | text-embedding-3-* only | Any value up to max (1536 or 3072) | Max dimension |
| **Cohere** | `output_dimension` | embed-v4+ only | 256, 512, 1024, 1536 (fixed set) | 1536 |
| **VoyageAI** | `output_dimension` | voyage-3-large, 3.5, code-3 | 256, 512, 1024, 2048 | 1024 |

OpenAI allows [any dimension up to the max](https://openai.com/index/new-embedding-models-and-api-updates/), while Cohere only accepts [specific preset values](https://docs.cohere.com/reference/embed). VoyageAI uses [the same parameter name as Cohere](https://docs.voyageai.com/docs/embeddings) but different valid values.

### Context Length Disparity

| Provider | Max Context |
|----------|-------------|
| OpenAI | 8,192 tokens |
| VoyageAI | 32,000 tokens |
| Cohere | 512 tokens (embed-v3) |

VoyageAI's voyage-3 supports [4x the context length of OpenAI](https://blog.voyageai.com/2024/09/18/voyage-3/), while Cohere's older models are limited to just 512 tokens.

### Output Type Support

| Provider | float | int8 | uint8 | binary | ubinary |
|----------|-------|------|-------|--------|---------|
| OpenAI | Yes | No | No | No | No |
| Cohere (v4) | Yes | Yes | Yes | Yes | Yes |
| VoyageAI (3.5) | Yes | Yes | Yes | Yes | Yes |

[Cohere embed-v4](https://docs.cohere.com/docs/cohere-embed) and [VoyageAI 3.5](https://blog.voyageai.com/2025/05/20/voyage-3-5/) support quantized output types for storage efficiency, but OpenAI doesn't offer this at all.

---

## Missing Basic Features

Most embedding clients lack features that should be standard.

### No Built-in Retry Logic

Proper retry handling requires [specific configuration](https://pypi.org/project/backoff/):

- Exponential backoff with jitter to avoid thundering herd
- Different handling for rate limits (retry with backoff) vs client errors (don't retry)
- Respecting `Retry-After` headers when present
- Maximum attempt limits to prevent infinite loops

Projects like Chonkie had to [manually add tenacity retry logic](https://github.com/chonkie-inc/chonkie/pull/198) for OpenAI embedding calls. This is boilerplate that every user must implement.

### Inconsistent Error Handling

Each provider returns errors differently:
- OpenAI raises `RateLimitError`, `APIError`, `Timeout`
- Cohere has different exception types
- VoyageAI may raise generic HTTP errors

There's no unified error taxonomy, forcing developers to catch provider-specific exceptions.

### No Usage Tracking

Built-in tracking for:
- Token counts per request
- Cost per operation
- Cumulative usage across sessions

...simply doesn't exist. Developers must manually track `usage.total_tokens` from responses and maintain their own cost calculations using pricing data gathered from multiple sources.

---

## No Single Source of Truth for Model Metadata

### Pricing Fragmentation

OpenAI shows [different prices on different documentation pages](https://community.openai.com/t/pricing-discrepancy-for-embedding-models-between-pricing-page-and-model-docs/1346972)—$0.065/1M tokens on one page, $0.13/1M on another for text-embedding-3-large.

Each provider maintains separate pricing pages with different formats:
- [OpenAI Pricing](https://platform.openai.com/docs/pricing)
- [VoyageAI Pricing](https://docs.voyageai.com/docs/pricing)
- [Cohere Pricing](https://cohere.com/pricing) (requires navigating to find embedding costs)

There's no standardized cost-per-token metric or central pricing database.

### Capability Discovery is Manual

To know what an embedding model supports, developers must:

1. Read provider-specific documentation
2. Check if dimension reduction is supported (and which parameter name)
3. Verify context length limits
4. Test which output types are available
5. Find rate limits (often undocumented or inconsistent)

### MTEB Doesn't Solve This

While the [MTEB benchmark](https://github.com/embeddings-benchmark/mteb) provides performance metrics, it recently had to [transition from self-reported to verified results](https://arxiv.org/html/2506.21182v1) due to reproducibility issues with self-reported model performance.

MTEB still doesn't provide:
- Pricing information
- Rate limits
- API-specific capabilities (dimension support, output types)
- Provider SDK quirks and known issues

---

## Conclusion

The embedding API ecosystem suffers from:

1. **Design neglect**: Major clients treat embeddings as secondary to chat completions
2. **SDK fragmentation**: Each provider's library has unique bugs and breaking changes
3. **Capability chaos**: No standard way to discover what features a model supports
4. **Missing fundamentals**: Retry logic, error handling, and usage tracking left to users
5. **Scattered metadata**: Pricing, limits, and capabilities spread across dozens of pages

This is why catsu exists—to provide a unified, well-tested client that handles all of this complexity behind a clean, consistent API.
