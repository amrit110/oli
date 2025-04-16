use anyhow::Result;
use oli_tui::communication::rpc::RpcServer;
use oli_tui::App;
use serde_json::json;
use std::sync::{Arc, Mutex};

fn main() -> Result<()> {
    // Initialize app state
    let app = Arc::new(Mutex::new(App::new()));

    // Set up RPC server
    let mut rpc_server = RpcServer::new();

    // Clone the app state for use in handlers
    let app_clone = app.clone();

    // Register method handlers
    rpc_server.register_method("query_model", move |params| {
        let mut app = app_clone.lock().unwrap();

        // Extract query from params
        let prompt = params["prompt"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing prompt parameter"))?;

        // Get model index if provided
        let model_index = params["model_index"].as_u64().unwrap_or(0) as usize;

        // Store the selected model index for the query
        app.log(&format!("Using model at index: {}", model_index));

        // Query the model with the selected model index
        match app.query_model(prompt) {
            Ok(response) => Ok(json!({ "response": response })),
            Err(err) => Err(anyhow::anyhow!("Error querying model: {}", err)),
        }
    });

    // Clone the app state for use in another handler
    let app_clone = app.clone();

    // Register method for getting available models
    rpc_server.register_method("get_available_models", move |_| {
        let app = app_clone.lock().unwrap();

        // Get available models
        let models = app
            .available_models
            .iter()
            .map(|m| {
                json!({
                    "name": m.name,
                    "description": m.description,
                    "supports_agent": m.has_agent_support()
                })
            })
            .collect::<Vec<_>>();

        Ok(json!({ "models": models }))
    });

    // Clone the app state for use in another handler
    let app_clone = app.clone();

    // Register method for getting tasks
    rpc_server.register_method("get_tasks", move |_| {
        let app = app_clone.lock().unwrap();
        Ok(json!({ "tasks": app.get_task_statuses() }))
    });

    // Run the RPC server
    println!("Starting Oli backend server");
    rpc_server.run()?;

    Ok(())
}
