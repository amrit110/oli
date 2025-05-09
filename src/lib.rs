pub mod agent;
pub mod apis;
pub mod app;
pub mod communication;
mod errors;
pub mod models;
pub mod prompts;
pub mod tools;

// Re-export key backend components
pub use agent::core::Agent;
pub use agent::core::LLMProvider;
pub use app::core::App;
pub use app::core::AppState;
pub use communication::rpc::RpcServer;
