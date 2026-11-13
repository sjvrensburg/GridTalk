//! Tool turns: a conversation in which the model may reply with a
//! tool call instead of an answer.
//!
//! [`ToolSession::turn`] takes the conversation so far and the tools
//! on offer and returns either calls to run or the answer. The
//! OpenAI-compatible backend sends the tools as function definitions
//! (the `tools` / `tool_calls` wire format). The local llama.cpp
//! backend has no such protocol, so it uses the text protocol in
//! this module: the tools are described in the system prompt, the
//! model replies with one JSON object (`{"tool": "get_cells",
//! "range": "Sheet1!A1:D4"}`) or with `ANSWER:` and its answer, and a
//! grammar ([`tool_grammar`]) keeps the reply to one of the two. The
//! protocol depends on no model's own tool template, so any chat
//! model can be driven by it.

use crate::{CancelToken, InferError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// One call the model asked for. `arguments` is the JSON object the
/// model wrote; when it wrote something that is not an object, the
/// raw text is kept as a JSON string so the caller can say what was
/// wrong with it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// An assistant message: the calls it made.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// A tool message: the id of the call it answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// A tool message: the tool that ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    fn plain(role: Role, content: impl Into<String>) -> Message {
        Message {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }
    pub fn system(content: impl Into<String>) -> Message {
        Message::plain(Role::System, content)
    }
    pub fn user(content: impl Into<String>) -> Message {
        Message::plain(Role::User, content)
    }
    /// The model's turn: what it said and the calls it made.
    pub fn assistant(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Message {
        Message {
            tool_calls,
            ..Message::plain(Role::Assistant, content)
        }
    }
    /// The result of `call`.
    pub fn tool(call: &ToolCall, content: impl Into<String>) -> Message {
        Message {
            tool_call_id: Some(call.id.clone()),
            name: Some(call.name.clone()),
            ..Message::plain(Role::Tool, content)
        }
    }
}

/// A tool on offer: its name, what it does, and the JSON Schema of
/// its arguments (an object schema with `properties` and `required`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// What the model may reply with this turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    /// A tool call or the answer.
    Auto,
    /// A tool call. A backend that cannot force one returns whatever
    /// the model wrote, so the caller still checks.
    Required,
    /// The answer. The tools stay described, so the prompt up to this
    /// turn is unchanged and a cached prefix is reused.
    None,
}

#[derive(Debug, Clone)]
pub struct TurnRequest<'a> {
    pub messages: &'a [Message],
    pub tools: &'a [ToolSpec],
    pub choice: ToolChoice,
    pub max_tokens: usize,
}

/// The model's reply: `calls` to run, or, when there are none, `text`
/// as the answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    pub text: String,
    pub calls: Vec<ToolCall>,
    /// What the turn cost, when the backend reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// Token counts of one turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// The whole prompt.
    pub prompt_tokens: usize,
    /// The part of the prompt the backend had already evaluated in
    /// an earlier turn of the session.
    pub reused_tokens: usize,
    pub generated_tokens: usize,
}

/// A conversation held open across turns, so a backend can keep what
/// it has already processed (the local backend keeps its context and
/// evaluates only the new part of each prompt).
pub trait ToolSession {
    /// One model turn. `on_token` receives the text of an answer as it
    /// is produced, and nothing for a turn that is a tool call.
    fn turn(
        &mut self,
        req: &TurnRequest,
        cancel: &CancelToken,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<Turn, InferError>;

    /// How many tokens the conversation may grow to, when the backend
    /// knows.
    fn context_tokens(&self) -> Option<usize> {
        None
    }
}

/// What an answer starts with in the text protocol.
pub const ANSWER_MARK: &str = "ANSWER:";

fn properties(spec: &ToolSpec) -> Vec<(&str, &str, bool)> {
    let required: Vec<&str> = spec.parameters["required"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    let mut props: Vec<(&str, &str, bool)> = spec.parameters["properties"]
        .as_object()
        .map(|map| {
            map.iter()
                .map(|(name, schema)| {
                    (
                        name.as_str(),
                        schema["type"].as_str().unwrap_or("string"),
                        required.contains(&name.as_str()),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    // Required arguments first, each group by name: the order the
    // grammar writes them in.
    props.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(b.0)));
    props
}

fn example_call(spec: &ToolSpec) -> String {
    let mut text = format!("{{\"tool\":{}", serde_json::Value::from(spec.name.clone()));
    for (name, kind, required) in properties(spec) {
        if required {
            let value = match kind {
                "integer" | "number" => "1",
                "boolean" => "true",
                _ => "\"...\"",
            };
            text.push_str(&format!(",{}:{value}", serde_json::Value::from(name)));
        }
    }
    text.push('}');
    text
}

/// The part of the system prompt that describes the tools and the
/// reply format of the text protocol.
pub fn tools_prompt(tools: &[ToolSpec]) -> String {
    let Some(first) = tools.first() else {
        return String::new();
    };
    let mut text = format!(
        "You can call tools. To call one, reply with one JSON object and nothing else, such as {}. You are then sent its result and can call another tool or answer. When you have read what the answer needs, reply with {ANSWER_MARK} followed by the answer.\n\nTools:",
        example_call(first)
    );
    for spec in tools {
        let args: Vec<String> = properties(spec)
            .iter()
            .map(|(name, kind, required)| {
                if *required {
                    format!("\"{name}\": {kind}")
                } else {
                    format!("\"{name}\": {kind} (optional)")
                }
            })
            .collect();
        text.push_str(&format!(
            "\n- {} {{{}}}: {}",
            spec.name,
            args.join(", "),
            spec.description
        ));
    }
    text
}

fn gbnf_literal(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A JSON string literal as a GBNF literal: `search` becomes
/// `"\"search\""`.
fn gbnf_json_string(name: &str) -> String {
    gbnf_literal(&serde_json::Value::from(name).to_string())
}

/// The GBNF grammar of one reply in the text protocol: a call to one
/// of `tools` with arguments of the declared types (required ones in
/// name order, then any optional ones), or `ANSWER:` and free text,
/// as `choice` allows. With no tools only an answer is possible.
pub fn tool_grammar(tools: &[ToolSpec], choice: ToolChoice) -> String {
    let answer = format!("answer ::= {} [^\\x00]*\n", gbnf_literal(ANSWER_MARK));
    if tools.is_empty() || choice == ToolChoice::None {
        return format!("root ::= answer\n{answer}");
    }
    let mut grammar = String::from(match choice {
        ToolChoice::Required => "root ::= call\n",
        _ => "root ::= call | answer\n",
    });
    grammar.push_str(&answer);
    let alternatives: Vec<String> = (0..tools.len()).map(|i| format!("call-{i}")).collect();
    grammar.push_str(&format!(
        "call ::= \"{{\" ws {} ws \":\" ws ({}) ws \"}}\"\n",
        gbnf_json_string("tool"),
        alternatives.join(" | ")
    ));
    for (i, spec) in tools.iter().enumerate() {
        let mut rule = format!("call-{i} ::= {}", gbnf_json_string(&spec.name));
        for (name, kind, required) in properties(spec) {
            let value = match kind {
                "integer" => "integer",
                "number" => "number",
                "boolean" => "boolean",
                _ => "string",
            };
            let member = format!("\",\" ws {} ws \":\" ws {value}", gbnf_json_string(name));
            if required {
                rule.push_str(&format!(" {member}"));
            } else {
                rule.push_str(&format!(" ({member})?"));
            }
        }
        grammar.push_str(&rule);
        grammar.push('\n');
    }
    grammar.push_str(concat!(
        "string ::= \"\\\"\" ([^\"\\\\\\x00-\\x1f] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* \"\\\"\"\n",
        "integer ::= \"-\"? (\"0\" | [1-9] [0-9]*)\n",
        "number ::= integer (\".\" [0-9]+)? ([eE] [-+]? [0-9]+)?\n",
        "boolean ::= \"true\" | \"false\"\n",
        "ws ::= \" \"?\n",
    ));
    grammar
}

/// Text after a reasoning block, for models that think aloud first.
fn after_thinking(text: &str) -> &str {
    match text.rfind("</think>") {
        Some(at) => &text[at + "</think>".len()..],
        None => text,
    }
}

/// An answer's text without the protocol's mark.
pub fn strip_answer_mark(text: &str) -> &str {
    let text = after_thinking(text).trim();
    text.strip_prefix(ANSWER_MARK).unwrap_or(text).trim_start()
}

/// Read one reply of the text protocol. A reply that starts with `{`
/// is a call: `{"tool": name, ...arguments}`. One that does not parse
/// is returned as a call with an empty name and the raw text as its
/// arguments, for the caller to send back as an error. Anything else
/// is the answer. `id` names the call.
pub fn parse_text_turn(text: &str, id: &str) -> Turn {
    let body = after_thinking(text).trim();
    if !body.starts_with('{') {
        return Turn {
            text: strip_answer_mark(text).to_string(),
            calls: Vec::new(),
            usage: None,
        };
    }
    // The first JSON value; a model without a grammar may add words
    // after it.
    let first = serde_json::Deserializer::from_str(body)
        .into_iter::<serde_json::Value>()
        .next();
    let call = match first {
        Some(Ok(serde_json::Value::Object(mut map))) => {
            match map
                .remove("tool")
                .and_then(|t| t.as_str().map(String::from))
            {
                Some(name) => ToolCall {
                    id: id.to_string(),
                    name,
                    arguments: serde_json::Value::Object(map),
                },
                None => ToolCall {
                    id: id.to_string(),
                    name: String::new(),
                    arguments: serde_json::Value::String(body.to_string()),
                },
            }
        }
        _ => ToolCall {
            id: id.to_string(),
            name: String::new(),
            arguments: serde_json::Value::String(body.to_string()),
        },
    };
    Turn {
        text: String::new(),
        calls: vec![call],
        usage: None,
    }
}

/// A call as the text protocol writes it: `{"tool": name, ...args}`.
pub fn call_text(call: &ToolCall) -> String {
    let mut text = format!("{{\"tool\":{}", serde_json::Value::from(call.name.clone()));
    if let Some(map) = call.arguments.as_object() {
        for (name, value) in map {
            text.push_str(&format!(
                ",{}:{value}",
                serde_json::Value::from(name.clone())
            ));
        }
    }
    text.push('}');
    text
}

/// The conversation as plain `(role, content)` messages for a chat
/// template that knows only `system`, `user` and `assistant`: the
/// tools are described at the end of the system message, a call is
/// the assistant's JSON object, and a tool result is a user message
/// (`Result of get_cells:` and the text). Neighbouring messages of
/// one role are joined, since several templates require the roles to
/// alternate.
pub fn text_messages(messages: &[Message], tools: &[ToolSpec]) -> Vec<(&'static str, String)> {
    let mut out: Vec<(&'static str, String)> = Vec::new();
    let mut push = |role: &'static str, content: String| match out.last_mut() {
        Some((last, text)) if *last == role => {
            text.push_str("\n\n");
            text.push_str(&content);
        }
        _ => out.push((role, content)),
    };
    let described = tools_prompt(tools);
    let mut tools_placed = described.is_empty();
    for message in messages {
        match message.role {
            Role::System => {
                let mut content = message.content.clone();
                if !tools_placed {
                    content.push_str("\n\n");
                    content.push_str(&described);
                    tools_placed = true;
                }
                push("system", content);
            }
            Role::User => {
                if !tools_placed {
                    push("system", described.clone());
                    tools_placed = true;
                }
                push("user", message.content.clone());
            }
            Role::Assistant => {
                let mut parts: Vec<String> = Vec::new();
                if !message.content.trim().is_empty() {
                    parts.push(message.content.clone());
                }
                parts.extend(message.tool_calls.iter().map(call_text));
                push("assistant", parts.join("\n"));
            }
            Role::Tool => push(
                "user",
                format!(
                    "Result of {}:\n{}",
                    message.name.as_deref().unwrap_or("the call"),
                    message.content
                ),
            ),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn specs() -> Vec<ToolSpec> {
        vec![
            ToolSpec {
                name: "search".into(),
                description: "Find regions.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["query"]
                }),
            },
            ToolSpec {
                name: "what_if".into(),
                description: "Try a value.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "cell": {"type": "string"},
                        "value": {"type": "number"},
                        "target": {"type": "string"}
                    },
                    "required": ["cell", "value", "target"]
                }),
            },
            ToolSpec {
                name: "workbook_info".into(),
                description: "Sheets and names.".into(),
                parameters: serde_json::json!({"type": "object", "properties": {}, "required": []}),
            },
        ]
    }

    #[test]
    fn the_grammar_names_each_tool_and_its_arguments() {
        let tools = specs();
        let auto = tool_grammar(&tools, ToolChoice::Auto);
        assert!(auto.starts_with("root ::= call | answer\n"), "{auto}");
        assert!(
            auto.contains("answer ::= \"ANSWER:\" [^\\x00]*\n"),
            "{auto}"
        );
        assert!(
            auto.contains("call ::= \"{\" ws \"\\\"tool\\\"\" ws \":\" ws (call-0 | call-1 | call-2) ws \"}\"\n"),
            "{auto}"
        );
        // Required arguments in name order, then the optional one.
        assert!(
            auto.contains(
                "call-0 ::= \"\\\"search\\\"\" \",\" ws \"\\\"query\\\"\" ws \":\" ws string (\",\" ws \"\\\"limit\\\"\" ws \":\" ws integer)?\n"
            ),
            "{auto}"
        );
        assert!(
            auto.contains(
                "call-1 ::= \"\\\"what_if\\\"\" \",\" ws \"\\\"cell\\\"\" ws \":\" ws string \",\" ws \"\\\"target\\\"\" ws \":\" ws string \",\" ws \"\\\"value\\\"\" ws \":\" ws number\n"
            ),
            "{auto}"
        );
        assert!(
            auto.contains("call-2 ::= \"\\\"workbook_info\\\"\"\n"),
            "{auto}"
        );
        assert!(tool_grammar(&tools, ToolChoice::Required).starts_with("root ::= call\n"));
        // No call is possible once the choice is `None`, or with no
        // tools at all.
        let none = tool_grammar(&tools, ToolChoice::None);
        assert!(
            none.starts_with("root ::= answer\n") && !none.contains("call"),
            "{none}"
        );
        assert_eq!(tool_grammar(&[], ToolChoice::Required), none);
    }

    #[test]
    fn a_reply_is_read_as_a_call_or_an_answer() {
        let call = parse_text_turn(
            " {\"tool\":\"search\",\"query\":\"rate per hour\",\"limit\":5}\n",
            "call_1",
        );
        assert_eq!(
            call.calls,
            vec![ToolCall {
                id: "call_1".into(),
                name: "search".into(),
                arguments: serde_json::json!({"query": "rate per hour", "limit": 5}),
            }]
        );
        // What the protocol writes back reads as the same call.
        assert_eq!(parse_text_turn(&call_text(&call.calls[0]), "call_1"), call);
        // Words after the object are dropped; thinking before it too.
        let wordy = parse_text_turn(
            "<think>look it up</think>{\"tool\":\"workbook_info\"} then I will answer",
            "c",
        );
        assert_eq!(wordy.calls[0].name, "workbook_info");
        assert_eq!(wordy.calls[0].arguments, serde_json::json!({}));
        // A cut-off object is a call the caller can report on.
        let cut = parse_text_turn("{\"tool\":\"search\",\"query\":\"ra", "c");
        assert_eq!(cut.calls[0].name, "");
        assert!(cut.calls[0].arguments.is_string());
        assert_eq!(parse_text_turn("{\"query\":\"x\"}", "c").calls[0].name, "");
        // Anything else is the answer, without its mark.
        let answer = parse_text_turn("ANSWER: Total is 6 (Sheet1!D2).", "c");
        assert!(answer.calls.is_empty());
        assert_eq!(answer.text, "Total is 6 (Sheet1!D2).");
        assert_eq!(parse_text_turn("It is 6.", "c").text, "It is 6.");
    }

    #[test]
    fn the_conversation_renders_for_a_plain_chat_template() {
        let tools = specs();
        let call = ToolCall {
            id: "call_1".into(),
            name: "search".into(),
            arguments: serde_json::json!({"query": "total"}),
        };
        let messages = vec![
            Message::system("Answer from the workbook."),
            Message::user("What is the total?"),
            Message::assistant("", vec![call.clone()]),
            Message::tool(&call, "[Sheet1!A1:D4] score 0.5"),
            Message::user("Go on."),
        ];
        let rendered = text_messages(&messages, &tools);
        let roles: Vec<&str> = rendered.iter().map(|(r, _)| *r).collect();
        assert_eq!(roles, ["system", "user", "assistant", "user"]);
        assert!(rendered[0]
            .1
            .starts_with("Answer from the workbook.\n\nYou can call tools."));
        assert!(
            rendered[0]
                .1
                .contains("such as {\"tool\":\"search\",\"query\":\"...\"}."),
            "{}",
            rendered[0].1
        );
        assert!(
            rendered[0].1.contains(
                "\n- search {\"query\": string, \"limit\": integer (optional)}: Find regions."
            ),
            "{}",
            rendered[0].1
        );
        assert!(rendered[0]
            .1
            .contains("\n- workbook_info {}: Sheets and names."));
        assert_eq!(rendered[2].1, "{\"tool\":\"search\",\"query\":\"total\"}");
        // The result and the next user message are one user turn.
        assert_eq!(
            rendered[3].1,
            "Result of search:\n[Sheet1!A1:D4] score 0.5\n\nGo on."
        );
        // With no tools the system message is left alone.
        assert_eq!(
            text_messages(&messages[..1], &[])[0].1,
            "Answer from the workbook."
        );
    }
}
