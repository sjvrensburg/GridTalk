//! A model's own Jinja chat template, rendered with minijinja.
//!
//! llama.cpp's built-in applier recognises a fixed list of template
//! families and refuses the rest (Gemma 4, Spark-X2.6), so the local
//! backend renders those through the template the GGUF carries, as
//! the Hugging Face tokenizer would. Only what the text protocol
//! sends is supplied: `messages` with `role` and `content` (system,
//! user, assistant), `add_generation_prompt`, `bos_token` and
//! `eos_token`; tools travel in the system message, so `tools` is
//! left undefined.

use minijinja::{Environment, Error, ErrorKind, Value};
use std::collections::BTreeMap;

/// `template` rendered for `messages`, ending where the assistant's
/// reply starts.
pub fn render(
    template: &str,
    messages: &[(&str, String)],
    bos_token: &str,
    eos_token: &str,
) -> Result<String, String> {
    let mut env = Environment::new();
    // Python's string methods (`.strip()`, `.startswith()`,
    // `.split()`), which Hugging Face templates call freely.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |message: String| -> Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        },
    );
    // Templates that stamp today's date into the system prompt.
    env.add_function("strftime_now", |_format: String| -> Value {
        Value::from("")
    });
    env.add_template("chat", template)
        .map_err(|e| format!("chat template does not parse: {e}"))?;
    let messages: Vec<Value> = messages
        .iter()
        .map(|(role, content)| {
            Value::from(BTreeMap::from([
                ("role", Value::from(*role)),
                ("content", Value::from(content.as_str())),
            ]))
        })
        .collect();
    let context = Value::from(BTreeMap::from([
        ("messages", Value::from(messages)),
        ("add_generation_prompt", Value::from(true)),
        ("bos_token", Value::from(bos_token)),
        ("eos_token", Value::from(eos_token)),
    ]));
    env.get_template("chat")
        .and_then(|t| t.render(context))
        .map_err(|e| format!("chat template: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn messages() -> Vec<(&'static str, String)> {
        vec![
            ("system", "Use the tools.".to_string()),
            ("user", "  What is the total?  ".to_string()),
        ]
    }

    /// A Gemma-style template: the system message folded into the
    /// first user turn, Python string methods, and the generation
    /// prompt at the end.
    #[test]
    fn renders_a_template_the_builtin_applier_does_not_know() {
        let template = concat!(
            "{{ bos_token }}",
            "{%- if messages[0]['role'] == 'system' -%}",
            "{%- set prefix = messages[0]['content'] + '\\n\\n' -%}{%- set rest = messages[1:] -%}",
            "{%- else -%}{%- set prefix = '' -%}{%- set rest = messages -%}{%- endif -%}",
            "{%- for m in rest -%}<|turn>{{ m['role'] }}\n{{ prefix if loop.first }}{{ m['content'].strip() }}<turn|>\n{% endfor -%}",
            "{%- if add_generation_prompt -%}<|turn>model\n{% endif -%}"
        );
        let prompt = render(template, &messages(), "<bos>", "<eos>").unwrap();
        assert_eq!(
            prompt,
            "<bos><|turn>user\nUse the tools.\n\nWhat is the total?<turn|>\n<|turn>model\n"
        );
    }

    /// `raise_exception` stops rendering with the template's message,
    /// and a template that does not parse says so.
    #[test]
    fn template_errors_are_reported() {
        let err = render(
            "{% if messages[0].role != 'user' %}{{ raise_exception('First message must be user') }}{% endif %}",
            &messages(),
            "",
            "",
        )
        .unwrap_err();
        assert!(err.contains("First message must be user"), "{err}");
        let err = render("{% for m in messages %}", &messages(), "", "").unwrap_err();
        assert!(err.starts_with("chat template does not parse"), "{err}");
    }
}
