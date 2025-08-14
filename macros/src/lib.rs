extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{quote, ToTokens};
use syn::{
    parse_macro_input, Ident, ItemFn, Pat, PatType, FnArg, Meta, Lit, Expr,
    punctuated::Punctuated, Token,
};
use syn::parse::Parse;

// Custom parsing for attribute arguments
struct AttributeArgs {
    attrs: Punctuated<Meta, Token![,]>,
}

impl Parse for AttributeArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(AttributeArgs {
            attrs: input.parse_terminated(Meta::parse, Token![,])?,
        })
    }
}

/// Procedural macro that transforms a Rust function into an LLM tool.
///
/// # Example
///
/// ```no_run
/// use merco_llmproxy::merco_tool;
///
/// #[merco_tool(description = "Calculates the sum of two integers")]
/// pub fn sum(a: i32, b: i32) -> i32 {
///     a + b
/// }
/// ```
///
/// This will automatically register the function as a tool that can be called by LLMs.
#[proc_macro_attribute]
pub fn merco_tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr_args = parse_macro_input!(attr as AttributeArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    // Extract function name and arguments
    let fn_name = input_fn.sig.ident.to_string();
    let fn_args: Vec<_> = input_fn
        .sig
        .inputs
        .iter()
        .filter_map(|arg| match arg {
            FnArg::Typed(PatType { pat, ty, .. }) => {
                if let Pat::Ident(pat_ident) = &**pat {
                    let arg_name = pat_ident.ident.to_string();
                    let arg_type = ty.to_token_stream().to_string();
                    Some((arg_name, arg_type))
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect();

    // Extract description from attribute
    let mut description = format!("Tool function: {}", fn_name);
    for meta in &attr_args.attrs {
        if let Meta::NameValue(name_value) = meta {
            if name_value.path.is_ident("description") {
                if let Expr::Lit(expr_lit) = &name_value.value {
                    if let Lit::Str(lit_str) = &expr_lit.lit {
                        description = lit_str.value();
                    }
                }
            }
        }
    }

    // Generate the tool struct name
    let tool_struct_name = Ident::new(&format!("{}ToolArgs", fn_name), Span::call_site());

    // Generate parameter properties for JsonSchema
    let param_properties = fn_args.iter().map(|(name, type_str)| {
        let type_json = match type_str.as_str().trim() {
            "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64" | "u128" | "usize" => "integer",
            "f32" | "f64" => "number",
            "String" | "& str" | "&'static str" => "string",
            "bool" => "boolean",
            _ => "object",
        };
        
        quote! {
            props.insert(#name.to_string(), ::serde_json::json!({ "type": #type_json }));
        }
    });

    // Generate required parameter names
    let required_params = fn_args.iter().map(|(name, _)| {
        quote! {
            #name.to_string()
        }
    });

    // Generate function wrapper fields
    let fn_ident = &input_fn.sig.ident;
    let arg_names: Vec<_> = fn_args.iter().map(|(name, _)| Ident::new(name, Span::call_site())).collect();
    let arg_structs = fn_args.iter().map(|(name, ty_str)| {
        let name_ident = Ident::new(name, Span::call_site());
        // Parse the type string back into a Type syn object for accurate quoting
        let syn_type: syn::Type = syn::parse_str(ty_str).unwrap_or_else(|_| panic!("Failed to parse type string: {}", ty_str));
        quote! {
            #name_ident: #syn_type
        }
    });

    // Generate automatic registration function name (internal use)
    let registration_fn = Ident::new(&format!("_register_{}_tool", fn_name), Span::call_site());

    // Generate the output code
    let expanded = quote! {
        // Include the original function
        #input_fn

        // Create the args struct
        #[derive(::serde::Deserialize)]
        struct #tool_struct_name {
            #(#arg_structs),*
        }

        // Create the tool definition function and automatic registration
        impl #tool_struct_name {
            fn __get_tool_definition() -> ::merco_llmproxy::traits::Tool {
                use ::merco_llmproxy::traits::JsonSchema;
                use ::serde_json::Map;

                let mut props = Map::new();
                #(#param_properties)*

                ::merco_llmproxy::traits::Tool {
                    name: #fn_name.to_string(),
                    description: #description.to_string(),
                    parameters: JsonSchema {
                        schema_type: "object".to_string(),
                        properties: Some(props),
                        required: Some(vec![#(#required_params),*]),
                    },
                }
            }

            // Execute the function with deserialized arguments
            fn __execute_impl(args_json: &str) -> ::std::result::Result<String, String> {
                match ::serde_json::from_str::<#tool_struct_name>(args_json) {
                    Ok(args) => {
                        // Call the original function using the deserialized arguments
                        let result = #fn_ident(#(args.#arg_names),*);
                        // Convert the function's result back to a JSON string
                        ::serde_json::to_string(&result)
                           .map_err(|e| format!("Failed to serialize result for {}: {}", #fn_name, e))
                    }
                    Err(e) => Err(format!("Failed to parse arguments for {}: {}", #fn_name, e)),
                }
            }
        }

        // Register the tool with the registry
        #[::ctor::ctor]
        fn #registration_fn() {
            let tool_def = #tool_struct_name::__get_tool_definition();
            ::merco_llmproxy::tools::__register_macro_tool(
                &#fn_name,
                tool_def,
                #tool_struct_name::__execute_impl,
            );
        }
    };

    TokenStream::from(expanded)
}
