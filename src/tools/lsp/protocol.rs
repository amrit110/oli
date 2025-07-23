use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize, Deserialize)]
pub struct RequestMessage {
    pub jsonrpc: String,
    pub id: RequestId,
    pub method: String,
    pub params: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    String(String),
    Number(u64),
}

impl Default for RequestId {
    fn default() -> Self {
        RequestId::Number(0)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResponseMessage {
    pub jsonrpc: String,
    #[serde(default)]
    pub id: RequestId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponseError>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResponseError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NotificationMessage {
    pub jsonrpc: String,
    pub method: String,
    pub params: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InitializeParams {
    pub process_id: Option<u32>,
    pub root_path: Option<String>,
    pub root_uri: Option<String>,
    pub initialization_options: Option<Value>,
    pub capabilities: ClientCapabilities,
    pub trace: Option<String>,
    pub workspace_folders: Option<Vec<WorkspaceFolder>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClientCapabilities {
    pub workspace: Option<WorkspaceClientCapabilities>,
    pub text_document: Option<TextDocumentClientCapabilities>,
    pub window: Option<WindowClientCapabilities>,
    pub experimental: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkspaceClientCapabilities {
    pub apply_edit: Option<bool>,
    pub workspace_edit: Option<WorkspaceEditCapability>,
    pub did_change_configuration: Option<DynamicRegistrationCapability>,
    pub did_change_watched_files: Option<DynamicRegistrationCapability>,
    pub symbol: Option<DynamicRegistrationCapability>,
    pub execute_command: Option<DynamicRegistrationCapability>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkspaceEditCapability {
    pub document_changes: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DynamicRegistrationCapability {
    pub dynamic_registration: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TextDocumentClientCapabilities {
    pub synchronization: Option<TextDocumentSyncClientCapabilities>,
    pub completion: Option<CompletionClientCapabilities>,
    pub hover: Option<DynamicRegistrationCapability>,
    pub signature_help: Option<DynamicRegistrationCapability>,
    pub declaration: Option<DynamicRegistrationCapability>,
    pub definition: Option<DynamicRegistrationCapability>,
    pub type_definition: Option<DynamicRegistrationCapability>,
    pub implementation: Option<DynamicRegistrationCapability>,
    pub references: Option<DynamicRegistrationCapability>,
    pub document_highlight: Option<DynamicRegistrationCapability>,
    pub document_symbol: Option<DynamicRegistrationCapability>,
    pub code_action: Option<DynamicRegistrationCapability>,
    pub code_lens: Option<DynamicRegistrationCapability>,
    pub document_link: Option<DynamicRegistrationCapability>,
    pub color_provider: Option<DynamicRegistrationCapability>,
    pub formatting: Option<DynamicRegistrationCapability>,
    pub range_formatting: Option<DynamicRegistrationCapability>,
    pub on_type_formatting: Option<DynamicRegistrationCapability>,
    pub rename: Option<DynamicRegistrationCapability>,
    pub folding_range: Option<DynamicRegistrationCapability>,
    pub selection_range: Option<DynamicRegistrationCapability>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TextDocumentSyncClientCapabilities {
    pub dynamic_registration: Option<bool>,
    pub will_save: Option<bool>,
    pub will_save_wait_until: Option<bool>,
    pub did_save: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionClientCapabilities {
    pub dynamic_registration: Option<bool>,
    pub completion_item: Option<CompletionItemCapability>,
    pub completion_item_kind: Option<CompletionItemKindCapability>,
    pub context_support: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionItemCapability {
    pub snippet_support: Option<bool>,
    pub commit_characters_support: Option<bool>,
    pub documentation_format: Option<Vec<MarkupKind>>,
    pub deprecated_support: Option<bool>,
    pub preselect_support: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionItemKindCapability {
    pub value_set: Option<Vec<u8>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WindowClientCapabilities {
    pub work_done_progress: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkspaceFolder {
    pub uri: String,
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MarkupKind {
    PlainText,
    Markdown,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TextDocumentItem {
    pub uri: String,
    pub language_id: String,
    pub version: u32,
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DidOpenTextDocumentParams {
    pub text_document: TextDocumentItem,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TextDocumentIdentifier {
    pub uri: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentSymbolParams {
    pub text_document: TextDocumentIdentifier,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SemanticTokensParams {
    pub text_document: TextDocumentIdentifier,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CodeLensParams {
    pub text_document: TextDocumentIdentifier,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TextDocumentPositionParams {
    pub text_document: TextDocumentIdentifier,
    pub position: crate::tools::lsp::models::Position,
}

pub fn get_initialize_params(root_path: &str) -> InitializeParams {
    InitializeParams {
        process_id: Some(std::process::id()),
        root_path: Some(root_path.to_string()),
        root_uri: Some(format!("file://{root_path}")),
        initialization_options: None,
        capabilities: ClientCapabilities {
            workspace: Some(WorkspaceClientCapabilities {
                apply_edit: Some(true),
                workspace_edit: Some(WorkspaceEditCapability {
                    document_changes: Some(true),
                }),
                did_change_configuration: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                did_change_watched_files: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                symbol: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                execute_command: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
            }),
            text_document: Some(TextDocumentClientCapabilities {
                synchronization: Some(TextDocumentSyncClientCapabilities {
                    dynamic_registration: Some(true),
                    will_save: Some(true),
                    will_save_wait_until: Some(true),
                    did_save: Some(true),
                }),
                completion: Some(CompletionClientCapabilities {
                    dynamic_registration: Some(true),
                    completion_item: Some(CompletionItemCapability {
                        snippet_support: Some(true),
                        commit_characters_support: Some(true),
                        documentation_format: Some(vec![MarkupKind::Markdown]),
                        deprecated_support: Some(true),
                        preselect_support: Some(true),
                    }),
                    completion_item_kind: Some(CompletionItemKindCapability {
                        value_set: Some(vec![
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                            21, 22, 23, 24, 25,
                        ]),
                    }),
                    context_support: Some(true),
                }),
                hover: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                signature_help: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                declaration: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                definition: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                type_definition: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                implementation: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                references: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                document_highlight: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                document_symbol: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                code_action: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                code_lens: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                document_link: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                color_provider: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                formatting: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                range_formatting: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                on_type_formatting: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                rename: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                folding_range: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
                selection_range: Some(DynamicRegistrationCapability {
                    dynamic_registration: Some(true),
                }),
            }),
            window: Some(WindowClientCapabilities {
                work_done_progress: Some(true),
            }),
            experimental: None,
        },
        trace: Some(String::from("off")),
        workspace_folders: Some(vec![WorkspaceFolder {
            uri: format!("file://{root_path}"),
            name: root_path
                .split('/')
                .next_back()
                .unwrap_or("workspace")
                .to_string(),
        }]),
    }
}
