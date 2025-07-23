use serde::{Deserialize, Serialize};

// Model name constants to avoid duplication
pub const ANTHROPIC_MODEL_NAME: &str = "claude-sonnet-4-20250514";
pub const OPENAI_MODEL_NAME: &str = "gpt-4o";
pub const GEMINI_MODEL_NAME: &str = "gemini-2.5-pro-exp-03-25";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub file_name: String,
    pub description: String,
    pub recommended_for: String,
    pub supports_agent: bool,
}

impl ModelConfig {
    pub fn has_agent_support(&self) -> bool {
        self.supports_agent
    }
}

use crate::apis::ollama::OllamaClient;
use anyhow::Result;

pub fn get_available_models() -> Vec<ModelConfig> {
    // Start with just the API models
    let mut models = vec![
        // Claude 4 Sonnet - Anthropic model supporting tool use
        ModelConfig {
            name: "Claude 4 Sonnet".into(),
            file_name: ANTHROPIC_MODEL_NAME.into(),
            description: "Latest Anthropic Claude with advanced code capabilities".into(),
            recommended_for: "Professional code tasks, requires ANTHROPIC_API_KEY".into(),
            supports_agent: true,
        },
        // GPT-4o - OpenAI model supporting tool use
        ModelConfig {
            name: "GPT-4o".into(),
            file_name: OPENAI_MODEL_NAME.into(),
            description: "Latest OpenAI model with advanced tool use capabilities".into(),
            recommended_for: "Professional code tasks, requires OPENAI_API_KEY".into(),
            supports_agent: true,
        },
        // Gemini 2.5 Pro - Google model supporting tool use
        ModelConfig {
            name: "Gemini 2.5 Pro".into(),
            file_name: GEMINI_MODEL_NAME.into(),
            description: "Google's latest Gemini model with advanced code capabilities".into(),
            recommended_for: "Professional code tasks, requires GEMINI_API_KEY".into(),
            supports_agent: true,
        },
    ];

    // Try to fetch available models from Ollama
    if let Ok(ollama_models) = get_available_ollama_models() {
        // Add each available Ollama model to the list
        for model_info in ollama_models {
            // Create a description based on the model details
            let description = if let Some(details) = &model_info.details {
                if let Some(desc) = &details.description {
                    format!("{desc} - Running locally via Ollama")
                } else {
                    format!("{} - Running locally via Ollama", model_info.name)
                }
            } else {
                format!("{} - Running locally via Ollama", model_info.name)
            };

            // Add the model to the list with "(local)" suffix
            models.push(ModelConfig {
                name: format!("{} (local)", model_info.name),
                file_name: model_info.name.clone(),
                description,
                recommended_for: "Local code tasks, requires Ollama to be running".into(),
                supports_agent: true,
            });
        }
    }

    models
}

fn get_available_ollama_models() -> Result<Vec<crate::apis::ollama::OllamaModelInfo>> {
    // Try to get the list of models from Ollama in a non-async context
    // We'll use a short timeout to avoid blocking the UI if Ollama is not running

    // Create a runtime for the async call
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    // Try to list models with a timeout - if it fails, we just return an empty list
    let result = runtime.block_on(async {
        // Handle the client creation result explicitly rather than using ?
        match OllamaClient::new(None) {
            Ok(ollama_client) => {
                // Use a short timeout (2 seconds) to avoid hanging if Ollama is not running
                match tokio::time::timeout(
                    std::time::Duration::from_secs(2),
                    ollama_client.list_models(),
                )
                .await
                {
                    Ok(models_result) => match models_result {
                        Ok(models) => {
                            eprintln!("Found {} Ollama models", models.len());
                            Ok(models)
                        }
                        Err(e) => {
                            eprintln!("Error listing Ollama models: {e}");
                            Err(anyhow::anyhow!("Failed to list Ollama models: {}", e))
                        }
                    },
                    Err(_) => {
                        eprintln!("Timeout waiting for Ollama - likely not running");
                        Err(anyhow::anyhow!("Timeout waiting for Ollama"))
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to create Ollama client: {e}");
                Err(anyhow::anyhow!("Failed to create Ollama client: {}", e))
            }
        }
    });

    // Return empty list on any error
    match result {
        Ok(models) => Ok(models),
        Err(e) => {
            eprintln!("Returning empty models list due to error: {e}");
            Ok(Vec::new())
        }
    }
}
