import React, { useEffect, useState, useCallback, useMemo } from "react";
import { Box } from "ink";
import { BackendService } from "../services/backend.js";
import ChatInterface from "./ChatInterface.js";
import ModelSelector from "./ModelSelector.js";
import StatusBar from "./StatusBar.js";
// Theme is used by imported components

import { AppState, ToolExecution, ToolStatusUpdate, Model } from "../types/index.js";
import { isCommand } from "../utils/commandUtils.js";
import {
  executeCommand,
  processUserMessage,
} from "../utils/commandHandlers.js";

// App props interface
interface AppProps {
  backend: BackendService;
  initialPrompt?: string | null;
  initialModelIndex?: number;
}

// Main app component
const App: React.FC<AppProps> = ({ backend, initialPrompt, initialModelIndex }) => {
  // App state
  const [state, setState] = useState<AppState>({
    models: [],
    selectedModel: initialModelIndex !== undefined ? initialModelIndex : 0,
    messages: [],
    isProcessing: false,
    error: null,
    backendConnected: false,
    appMode: initialModelIndex !== undefined ? "chat" : "setup", // Only skip setup if model explicitly provided
    useAgent: true, // Agent mode is always enabled
  });

  // Tool executions state - separate to avoid re-rendering the entire app on tool updates
  const [toolExecutions, setToolExecutions] = useState<
    Map<string, ToolExecution>
  >(new Map());

  // Track if initial prompt has been processed
  const [initialPromptProcessed, setInitialPromptProcessed] = useState<boolean>(false);

  // UI state
  const [showShortcuts, setShowShortcuts] = useState(false);

  // Subscribe to tool status events
  useEffect(() => {
    // Setup tool status subscription when backend is available
    const setupToolStatusSubscription = async () => {
      try {
        await backend.subscribe("tool_status");
        // Subscribed successfully
      } catch (error) {
        console.error("Failed to subscribe to tool status updates:", error);
      }
    };

    // Handle tool status events
    const handleToolStatus = (params: ToolStatusUpdate) => {
      const { type, execution } = params;

      setToolExecutions((prev) => {
        // Create a new Map to avoid mutating state
        const newMap = new Map(prev);

        if (type === "started") {
          // Add new tool execution to the map
          newMap.set(execution.id, execution);
        } else if (type === "updated") {
          // Update existing tool in the map
          newMap.set(execution.id, execution);

          // When a tool completes, add a message to the chat history
          if (execution.status !== "running" && execution.endTime) {
            setState((prev) => {
              // Add a tool result message to the messages array
              return {
                ...prev,
                messages: [
                  ...prev.messages,
                  {
                    id: `tool-result-${execution.id}`,
                    role: "tool",
                    content: `[${execution.name}] ${execution.message}`,
                    timestamp: Date.now(),
                    task_id: execution.task_id,
                    tool: execution.name,
                    tool_status:
                      execution.status === "success" ? "success" : "error",
                    tool_data: {
                      name: execution.name,
                      file_path: execution.metadata.file_path as
                        | string
                        | undefined,
                      lines: execution.metadata.lines as number | undefined,
                      description:
                        execution.message ||
                        (execution.metadata.description as string | undefined),
                      // Pass through the full metadata to ensure pattern is available
                      metadata: execution.metadata,
                    },
                  },
                ],
              };
            });

            // Remove completed tool from the map after a short delay
            setTimeout(() => {
              setToolExecutions((current) => {
                const updatedMap = new Map(current);
                updatedMap.delete(execution.id);
                return updatedMap;
              });
            }, 3000);
          }
        }

        return newMap;
      });
    };

    // Subscribe when component mounts
    backend.on("tool_status", handleToolStatus);
    setupToolStatusSubscription();

    // Unsubscribe when component unmounts
    return () => {
      backend.off("tool_status", handleToolStatus);
      backend.unsubscribe("tool_status").catch(console.error);
    };
  }, [backend]);

  // Load initial data
  useEffect(() => {
    // Check if backend is already connected via the flag
    if ((backend as any).isConnected) {
      // If already connected, fetch models directly
      backend.call("get_available_models").then(result => {
        const models = (result.models as Model[]) || [];

        setState((prev) => ({
          ...prev,
          models: models,
          backendConnected: true,
          // If we have an initial prompt or model, go straight to chat mode
          ...(initialPrompt || initialModelIndex !== undefined ? { appMode: "chat" } : {}),
        }));
      }).catch(err => {
        });
    }

    // Listen for backend connection events
    backend.on("backend_connected", async (params) => {
      setState((prev) => ({
        ...prev,
        models: params.models || [],
        backendConnected: true,
        backendInfo: {
          ...params,
        },
        // If we have an initial prompt or model, go straight to chat mode
        ...(initialPrompt || initialModelIndex !== undefined ? { appMode: "chat" } : {}),
      }));

      // If an initial model was specified, set it in the backend
      if (initialModelIndex !== undefined && initialModelIndex !== 0) {
        try {
          await backend.call("set_selected_model", { model_index: initialModelIndex });
        } catch (error) {
          // Failed to set initial model, but continue anyway
        }
      }

      // If there's an initial prompt and we're now connected, process it
      if (initialPrompt && !initialPromptProcessed) {
        // Skip model selection and go straight to chat with initial prompt
        setInitialPromptProcessed(true);

        // Add the user message to the chat history
        setState((prev) => ({
          ...prev,
          appMode: "chat", // Switch to chat mode immediately
          messages: [
            ...prev.messages,
            {
              id: `user-${Date.now()}`,
              role: "user",
              content: initialPrompt,
              timestamp: Date.now(),
            },
          ],
        }));

        // Process the initial prompt with the selected model
        backend.call("run", {
          prompt: initialPrompt,
          model_index: initialModelIndex
        }).catch((error) => {
          // Failed to process initial prompt, but continue
        });
      }
    });

    backend.on("backend_connection_error", (params) => {
      setState((prev) => ({
        ...prev,
        error: params.error,
        backendConnected: false,
        messages: [
          ...prev.messages,
          {
            id: `system-${Date.now()}`,
            role: "system",
            content: `Failed to connect to backend: ${params.error}`,
            timestamp: Date.now(),
          },
        ],
      }));
    });

    // Register event listeners for backend notifications
    backend.on("processing_started", (params) => {
      setState((prev) => ({
        ...prev,
        isProcessing: true,
        // If agent mode is specified in the event, update state
        ...(params.use_agent !== undefined
          ? { useAgent: params.use_agent }
          : {}),
      }));
    });

    backend.on("processing_progress", (params) => {
      // Add progress message if it's not already in the list
      setState((prev) => {
        // Only add the message if it's not a duplicate
        if (!prev.messages.some((m) => m.content === params.message)) {
          return {
            ...prev,
            messages: [
              ...prev.messages,
              {
                id: `progress-${Date.now()}`,
                role: "system",
                content: params.message,
                timestamp: Date.now(),
                task_id: params.task_id,
              },
            ],
          };
        }
        return prev;
      });
    });

    backend.on("processing_complete", () => {
      setState((prev) => ({
        ...prev,
        isProcessing: false,
      }));
    });

    backend.on("processing_error", (params) => {
      setState((prev) => ({
        ...prev,
        isProcessing: false,
        error: params.error,
        messages: [
          ...prev.messages,
          {
            id: `error-${Date.now()}`,
            role: "system",
            content: `Error: ${params.error}`,
            timestamp: Date.now(),
          },
        ],
      }));
    });

    // Handle legacy tool execution events by converting them to the new format
    backend.on("tool_execution", (params) => {
      // Generate a unique identifier for this tool execution
      const toolId = `tool-${params.tool}-${Date.now()}`;

      // Bridge old tool_execution events to the new tool_status system
      setToolExecutions((prev) => {
        const newMap = new Map(prev);
        const execution: ToolExecution = {
          id: toolId,
          task_id: params.task_id || "",
          name: params.tool,
          status: params.status || "running",
          startTime: Date.now(),
          endTime: params.status !== "running" ? Date.now() : undefined,
          message: params.message,
          metadata: {
            file_path: params.file_path,
            lines: params.lines,
            description: params.description,
          },
        };

        // Add to tool executions map
        newMap.set(toolId, execution);

        return newMap;
      });

      // Add a message to the state for the tool execution
      setState((prev) => {
        return {
          ...prev,
          messages: [
            ...prev.messages,
            {
              id: toolId,
              role: "tool",
              content: `[${params.tool}] ${params.message}`,
              timestamp: Date.now(),
              task_id: params.task_id,
              tool: params.tool,
              tool_status: params.status || "running",
              tool_data: {
                name: params.tool,
                file_path: params.file_path,
                lines: params.lines,
                description: params.description,
                metadata: {
                  file_path: params.file_path,
                  lines: params.lines,
                  description: params.description,
                  pattern: params.pattern,
                },
              },
            },
          ],
          // Task tracking is now handled through toolExecutions Map
        };
      });

      // If the tool is now complete, remove it from active tools after a delay
      if (params.status && params.status !== "running") {
        setTimeout(() => {
          setToolExecutions((current) => {
            const updatedMap = new Map(current);
            updatedMap.delete(toolId);
            return updatedMap;
          });
        }, 3000);
      }
    });

    backend.on("log_message", () => {
      // Silent log handling
    });

    // Clean up event listeners on component unmount
    return () => {
      backend.removeAllListeners();
    };
  }, [backend, initialPrompt, initialPromptProcessed, initialModelIndex]);

  // Handle model selection - memoized to prevent unnecessary rerenders
  const handleModelSelect = useCallback(
    (index: number) => {
      // Update local state for the UI
      setState((prev) => ({
        ...prev,
        selectedModel: index,
      }));

      // Notify the backend about the model change if connected
      if (backend) {
        backend
          .call("set_selected_model", { model_index: index })
          .catch((error) => {
            console.error("Failed to update selected model on backend:", error);
          });
      }
    },
    [backend],
  );

  // Memoize the clear history handler
  const handleClearHistory = useCallback(() => {
    // Clear all messages from the UI state
    setState((prev) => ({
      ...prev,
      messages: [], // Clear all messages
      error: null, // Also clear any error state
    }));
  }, []);

  // Memoize command execution handler to reduce rerenders
  const handleExecuteCommand = useCallback(
    (command: string) => {
      // First try to execute as a built-in command
      const wasHandled = executeCommand(command, state, setState, backend, {
        handleClearHistory,
        handleModelSelect,
      });

      // If not a built-in command, handle as regular input
      if (!wasHandled) {
        processUserMessage(command, state, setState, backend);
      }
    },
    [state, backend, handleClearHistory, handleModelSelect],
  );

  // Handle regular user input (non-commands)
  const handleRegularInput = useCallback(
    async (input: string) => {
      // Process user message without command handling
      await processUserMessage(input, state, setState, backend);
    },
    [state, setState, backend],
  );

  // Combined handler for all user input
  const handleUserInput = useCallback(
    async (input: string) => {
      // If this is a command, handle it separately through the command handler
      if (isCommand(input)) {
        handleExecuteCommand(input);
        return;
      }

      // This is a regular user message - send it to the backend
      await handleRegularInput(input);
    },
    [handleExecuteCommand, handleRegularInput],
  );

  // Handle model confirmation and switch to chat mode - memoized to prevent unnecessary rerenders
  const handleModelConfirm = useCallback(() => {
    // Only proceed if we have models and backend is connected
    if (state.models.length > 0 && state.backendConnected) {
      // Send the final model selection to the backend
      backend
        .call("set_selected_model", { model_index: state.selectedModel })
        .then(() => {
          // Then switch to chat mode
          setState((prev) => ({
            ...prev,
            appMode: "chat",
          }));
        })
        .catch((error) => {
          console.error("Failed to set model on backend:", error);
          // Still switch to chat mode even if there was an error
          setState((prev) => ({
            ...prev,
            appMode: "chat",
          }));
        });
    }
  }, [state.models, state.backendConnected, state.selectedModel, backend]);

  // Memoize the toggle shortcuts handler
  const handleToggleShortcuts = useCallback(() => {
    setShowShortcuts((prev) => !prev);
  }, []);

  // Memoize components to prevent unnecessary rerenders
  const modelSelectorComponent = useMemo(
    () => (
      <ModelSelector
        models={state.models}
        selectedIndex={state.selectedModel}
        onSelect={handleModelSelect}
        onConfirm={handleModelConfirm}
        isLoading={!state.backendConnected || state.models.length === 0}
      />
    ),
    [
      state.models,
      state.selectedModel,
      state.backendConnected,
      handleModelSelect,
      handleModelConfirm,
    ],
  );

  // Handle task interruption
  const handleInterrupt = useCallback(() => {
    // Call the backend to interrupt the current task
    if (state.isProcessing) {
      backend
        .call("interrupt_processing", {})
        .then(() => {
          setState((prev) => ({
            ...prev,
            isProcessing: false,
            messages: [
              ...prev.messages,
              {
                id: `system-${Date.now()}`,
                role: "system",
                content: "Task interrupted by user",
                timestamp: Date.now(),
              },
            ],
          }));
        })
        .catch((err) => {
          console.error("Failed to interrupt task:", err);
          // Set processing to false anyway to update UI
          setState((prev) => ({
            ...prev,
            isProcessing: false,
            messages: [
              ...prev.messages,
              {
                id: `system-${Date.now()}`,
                role: "system",
                content: "Attempted to interrupt task but encountered an error",
                timestamp: Date.now(),
              },
            ],
          }));
        });
    }
  }, [state.isProcessing, backend]);

  // Clean up message history to prevent duplicates
  const filteredMessages = useMemo(() => {
    // Track seen user messages to remove duplicates
    const seenUserMessages = new Set<string>();

    // Filter for a clean chat history
    return state.messages.filter((msg) => {
      // Keep all assistant messages
      if (msg.role === "assistant") return true;

      // For user messages, check for duplicates
      if (msg.role === "user") {
        // Skip duplicates based on content
        if (seenUserMessages.has(msg.content)) {
          return false;
        }

        // Mark as seen and keep
        seenUserMessages.add(msg.content);
        return true;
      }

      // For tools and system messages, keep them all
      return true;
    });
  }, [state.messages]);

  const chatInterfaceComponent = useMemo(
    () => (
      <ChatInterface
        messages={filteredMessages}
        isProcessing={state.isProcessing}
        onSubmit={handleUserInput}
        onInterrupt={handleInterrupt}
        showShortcuts={showShortcuts}
        onToggleShortcuts={handleToggleShortcuts}
        onClearHistory={handleClearHistory}
        onExecuteCommand={handleExecuteCommand}
        toolExecutions={toolExecutions}
      />
    ),
    [
      filteredMessages,
      state.isProcessing,
      toolExecutions,
      handleUserInput,
      handleInterrupt,
      showShortcuts,
      handleToggleShortcuts,
      handleClearHistory,
      handleExecuteCommand,
    ],
  );

  const statusBarComponent = useMemo(
    () => (
      <StatusBar
        modelName={state.models[state.selectedModel]?.name || "AI Assistant"}
        isProcessing={state.isProcessing}
        backendConnected={state.backendConnected}
        showShortcuts={showShortcuts}
      />
    ),
    [
      state.models,
      state.selectedModel,
      state.isProcessing,
      state.backendConnected,
      showShortcuts,
    ],
  );

  // Render with memoized components for better performance
  if (state.appMode === "setup") {
    // Setup mode - directly render the model selector without any container
    return modelSelectorComponent;
  }

  // Single column layout with component-based architecture
  return (
    <Box flexDirection="column" width="100%" height="100%">
      {/* Chat area */}
      <Box flexGrow={1} flexDirection="column">
        {chatInterfaceComponent}
      </Box>

      {/* Status bar */}
      {statusBarComponent}
    </Box>
  );
};

export default App;
