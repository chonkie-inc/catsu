//! Example usage of catsu with Mixedbread embeddings.

use catsu::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new()?;

    if !client.has_provider("mixedbread") {
        println!("Mixedbread provider not available. Set MIXEDBREAD_API_KEY environment variable.");
        return Ok(());
    }

    let response = client
        .embed(
            "mixedbread:mxbai-embed-large-v1",
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
