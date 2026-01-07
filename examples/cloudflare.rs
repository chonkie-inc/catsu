//! Example usage of catsu with Cloudflare Workers AI embeddings.

use catsu::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new()?;

    if !client.has_provider("cloudflare") {
        println!("Cloudflare provider not available.");
        println!("Set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID environment variables.");
        return Ok(());
    }

    let response = client
        .embed(
            "cloudflare:@cf/baai/bge-base-en-v1.5",
            vec!["Hello, world!".to_string(), "How are you?".to_string()],
        )
        .await?;

    println!("Model: {}", response.model);
    println!("Provider: {}", response.provider);
    println!("Dimensions: {}", response.dimensions);
    println!("Input count: {}", response.input_count);
    println!("Latency: {:.2}ms", response.latency_ms);
    println!("Total tokens: {}", response.usage.tokens);
    println!(
        "First embedding (first 5 dims): {:?}",
        &response.embeddings[0][..5]
    );

    Ok(())
}
