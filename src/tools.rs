use crate::traits::{Tool, ToolCallFunction};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use lazy_static::lazy_static;

/// Represents a tool function that can be executed with JSON arguments
pub type ToolExecutor = Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync>;

/// A registry for storing and managing tool functions
pub struct ToolRegistry {
    tools: HashMap<String, (Tool, ToolExecutor)>,
}

impl ToolRegistry {
    /// Create a new empty tool registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool with its tool definition and executor function
    pub fn register(&mut self, tool: Tool, executor: ToolExecutor) {
        self.tools.insert(tool.name.clone(), (tool, executor));
    }

    /// Get all registered tool definitions
    pub fn get_tools(&self) -> Vec<Tool> {
        self.tools.values().map(|(tool, _)| tool.clone()).collect()
    }

    /// Execute a tool by name with the provided arguments
    pub fn execute_tool(&self, name: &str, args: &str) -> Result<String, String> {
        match self.tools.get(name) {
            Some((_, executor)) => executor(args),
            None => Err(format!("Tool '{}' not found in registry", name)),
        }
    }

    /// Execute a tool call
    pub fn execute_tool_call(&self, tool_call: &ToolCallFunction) -> Result<String, String> {
        self.execute_tool(&tool_call.name, &tool_call.arguments)
    }
}

// Global registry singleton
lazy_static! {
    static ref GLOBAL_REGISTRY: Arc<Mutex<ToolRegistry>> = Arc::new(Mutex::new(ToolRegistry::new()));
}

/// Register a tool in the global registry
pub fn register_tool(tool: Tool, executor: ToolExecutor) {
    if let Ok(mut registry) = GLOBAL_REGISTRY.lock() {
        registry.register(tool, executor);
    } else {
        eprintln!("[Tool Registry] Failed to lock registry for registering tool.");
    }
}

/// Helper function for procedural macro to register a tool with tool definition and executor
#[doc(hidden)]
pub fn __register_macro_tool(tool_name: &str, tool_definition: Tool, executor_fn: impl Fn(&str) -> Result<String, String> + Send + Sync + 'static) {
    register_tool(tool_definition, Arc::new(executor_fn));
}

/// Get all registered tools from the global registry
pub fn get_all_tools() -> Vec<Tool> {
    GLOBAL_REGISTRY
        .lock()
        .map(|registry| registry.get_tools())
        .unwrap_or_default()
}

/// Get a specific subset of registered tools by their names.
///
/// # Arguments
///
/// * `names` - A slice of string slices representing the names of the tools to retrieve.
///
/// # Returns
///
/// A `Vec<Tool>` containing the definitions of the found tools. Tools not found in the
/// registry are silently ignored.
pub fn get_tools_by_names(names: &[&str]) -> Vec<Tool> {
    let registry = match GLOBAL_REGISTRY.lock() {
        Ok(reg) => reg,
        Err(e) => {
            eprintln!("[Tool Registry] Failed to lock registry for getting tools by name: {}", e);
            return Vec::new(); // Return empty list on lock failure
        }
    };

    names
        .iter()
        .filter_map(|name| registry.tools.get(*name).map(|(tool, _)| tool.clone()))
        .collect()
}

/// Execute a tool by name with JSON arguments
pub fn execute_tool(name: &str, args: &str) -> Result<String, String> {
    GLOBAL_REGISTRY
        .lock()
        .map_err(|e| format!("Failed to lock registry: {}", e))?
        .execute_tool(name, args)
}

/// Create a public re-export macro for the merco_tool attribute
#[cfg(feature = "macros")]
pub use merco_macros::merco_tool;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::JsonSchema;

    #[test]
    fn test_tool_registry() {
        let mut registry = ToolRegistry::new();
        
        // Create a simple addition tool
        let add_tool = Tool {
            name: "add".to_string(),
            description: "Add two numbers".to_string(),
            parameters: JsonSchema {
                schema_type: "object".to_string(),
                properties: Some({
                    let mut props = serde_json::Map::new();
                    props.insert("a".to_string(), serde_json::json!({"type": "integer"}));
                    props.insert("b".to_string(), serde_json::json!({"type": "integer"}));
                    props
                }),
                required: Some(vec!["a".to_string(), "b".to_string()]),
            },
        };
        
        // Create executor function
        let add_executor: ToolExecutor = Arc::new(|args| {
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(args);
            match parsed {
                Ok(value) => {
                    let a = value.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
                    let b = value.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
                    Ok((a + b).to_string())
                }
                Err(e) => Err(format!("Failed to parse arguments: {}", e)),
            }
        });
        
        // Register the tool
        registry.register(add_tool.clone(), add_executor);
        
        // Check that the tool was registered
        assert_eq!(registry.get_tools().len(), 1);
        assert_eq!(registry.get_tools()[0].name, "add");
        
        // Execute the tool
        let result = registry.execute_tool("add", r#"{"a": 5, "b": 3}"#);
        assert_eq!(result, Ok("8".to_string()));
        
        // Try executing a non-existent tool
        let error = registry.execute_tool("multiply", r#"{"a": 5, "b": 3}"#);
        assert!(error.is_err());
    }
} 