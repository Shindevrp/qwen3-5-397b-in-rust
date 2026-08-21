//! Qwen3.5 chat template — text-only subset of the official Jinja template
//! (`chat_template.jinja`), rendered as plain ChatML.
//!
//! Format (per message): `<|im_start|>{role}\n{content}<|im_end|>\n`
//!
//! Deviations from full Jinja are deliberate: no tools, no vision, no
//! `tool_response` wrapping. Everything else — system-message placement,
//! think-block splitting for assistant history, the `last_query_index`
//! rule (assistant turns after the last user query keep their reasoning),
//! and the generation prompt with `<think>` prefill — mirrors the official
//! template exactly.

/// Conversation roles supported by the text-only template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// A single chat message.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into() }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }
}

/// Rendering options mirroring the Jinja template's control variables.
#[derive(Debug, Clone, Copy)]
pub struct ChatRenderOptions {
    /// Append `<|im_start|>assistant\n<think>\n` so the model continues the
    /// assistant turn.
    pub add_generation_prompt: bool,
    /// `enable_thinking` in the official template. When false, an empty
    /// `<think>\n\n</think>\n\n` block is inserted after the generation
    /// prompt to skip reasoning entirely.
    pub enable_thinking: bool,
}

impl Default for ChatRenderOptions {
    fn default() -> Self {
        Self { add_generation_prompt: true, enable_thinking: true }
    }
}

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const THINK_CLOSE: &str = "</think>";

/// Split assistant content into `(reasoning, content)` following the
/// official template:
///
/// ```jinja
/// {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
/// {%- set content = content.split('</think>')[-1].lstrip('\n') %}
/// ```
///
/// i.e. reasoning is the text before the *first* `</think>` (after the last
/// `<think>` within it); content is everything after the *last* `</think>`.
/// If no `</think>` is present, all text is content and reasoning is empty.
fn split_reasoning(content: &str) -> (&str, &str) {
    if let Some(first_close) = content.find(THINK_CLOSE) {
        let mut reasoning = &content[..first_close];
        reasoning = reasoning.trim_end_matches('\n');
        if let Some(pos) = reasoning.rfind("<think>") {
            reasoning = &reasoning[pos + "<think>".len()..];
        }
        let reasoning = reasoning.trim_start_matches('\n');

        let body = match content.rfind(THINK_CLOSE) {
            Some(last_close) => &content[last_close + THINK_CLOSE.len()..],
            None => "",
        };
        let body = body.trim_start_matches('\n');

        (reasoning.trim(), body)
    } else {
        ("", content)
    }
}

/// Render a conversation into a single prompt string, exactly as the
/// official Qwen3.5 template would for text-only messages.
pub fn render_chat(messages: &[Message], opts: &ChatRenderOptions) -> Result<String, String> {
    if messages.is_empty() {
        return Err("No messages provided.".to_string());
    }

    let mut out = String::new();

    // System message must be first; its content is trimmed.
    let mut start = 0;
    if messages[0].role == Role::System {
        out.push_str(IM_START);
        out.push_str("system\n");
        out.push_str(messages[0].content.trim());
        out.push_str(IM_END);
        out.push('\n');
        start = 1;
    }

    // Index of the last user query. Assistant turns after it keep their
    // reasoning (`<think>` block); earlier ones have reasoning stripped.
    let last_query_index = messages
        .iter()
        .rposition(|m| m.role == Role::User)
        .ok_or_else(|| "No user query found in messages.".to_string())?;

    for (i, msg) in messages.iter().enumerate().skip(start) {
        match msg.role {
            Role::System => {
                return Err("System message must be at the beginning.".to_string());
            }
            Role::User => {
                out.push_str(IM_START);
                out.push_str("user\n");
                out.push_str(msg.content.trim());
                out.push_str(IM_END);
                out.push('\n');
            }
            Role::Assistant => {
                let trimmed = msg.content.trim();
                let (reasoning, content) = split_reasoning(trimmed);
                out.push_str(IM_START);
                out.push_str("assistant\n");
                if i > last_query_index {
                    out.push_str("<think>\n");
                    out.push_str(reasoning);
                    out.push_str("\n</think>\n\n");
                    out.push_str(content);
                } else {
                    out.push_str(content);
                }
                out.push_str(IM_END);
                out.push('\n');
            }
        }
    }

    if opts.add_generation_prompt {
        out.push_str(IM_START);
        out.push_str("assistant\n<think>\n");
        if !opts.enable_thinking {
            out.push_str("\n\n</think>\n\n");
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_user_generation_prompt_thinking() {
        let msgs = vec![
            Message::system("You are helpful."),
            Message::user("Hi there!"),
        ];
        let out = render_chat(&msgs, &ChatRenderOptions::default()).unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n\
             <|im_start|>user\nHi there!<|im_end|>\n\
             <|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn generation_prompt_thinking_disabled() {
        let msgs = vec![Message::user("Hi")];
        let out = render_chat(
            &msgs,
            &ChatRenderOptions { add_generation_prompt: true, enable_thinking: false },
        )
        .unwrap();
        assert_eq!(
            out,
            "<|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n\n</think>\n\n"
        );
    }

    #[test]
    fn no_system_message() {
        let msgs = vec![Message::user("Hello")];
        let out = render_chat(&msgs, &ChatRenderOptions::default()).unwrap();
        assert!(out.starts_with("<|im_start|>user\n"));
    }

    #[test]
    fn user_content_is_trimmed() {
        let msgs = vec![Message::user("  padded  \n")];
        let out = render_chat(&msgs, &ChatRenderOptions::default()).unwrap();
        assert_eq!(
            out,
            "<|im_start|>user\npadded<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn multi_turn_history_strips_old_reasoning() {
        // Assistant turn BEFORE the last user query: reasoning stripped.
        let msgs = vec![
            Message::user("What is 2+2?"),
            Message::assistant("<think>\nIt's trivial.\n</think>\n\n4"),
            Message::user("And 3+3?"),
        ];
        let out = render_chat(&msgs, &ChatRenderOptions::default()).unwrap();
        assert_eq!(
            out,
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n\
             <|im_start|>assistant\n4<|im_end|>\n\
             <|im_start|>user\nAnd 3+3?<|im_end|>\n\
             <|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn assistant_after_last_query_keeps_reasoning() {
        // Full conversation re-render (add_generation_prompt=false), as when
        // feeding the completed exchange back for the next turn.
        let msgs = vec![
            Message::user("Q1"),
            Message::assistant("<think>\nbecause\n</think>\n\nA1"),
            Message::user("Q2"),
            Message::assistant("A2"),
        ];
        let out = render_chat(
            &msgs,
            &ChatRenderOptions { add_generation_prompt: false, enable_thinking: true },
        )
        .unwrap();
        assert_eq!(
            out,
            "<|im_start|>user\nQ1<|im_end|>\n\
             <|im_start|>assistant\nA1<|im_end|>\n\
             <|im_start|>user\nQ2<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\nA2<|im_end|>\n"
        );
    }

    #[test]
    fn assistant_without_think_block() {
        let msgs = vec![
            Message::user("Q"),
            Message::assistant("Plain answer"),
        ];
        let out = render_chat(
            &msgs,
            &ChatRenderOptions { add_generation_prompt: false, enable_thinking: true },
        )
        .unwrap();
        assert_eq!(
            out,
            "<|im_start|>user\nQ<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\nPlain answer<|im_end|>\n"
        );
    }

    #[test]
    fn errors_on_empty_and_missing_user() {
        assert!(render_chat(&[], &ChatRenderOptions::default()).is_err());

        let only_assistant = vec![Message::assistant("hi")];
        let err = render_chat(&only_assistant, &ChatRenderOptions::default()).unwrap_err();
        assert_eq!(err, "No user query found in messages.");
    }

    #[test]
    fn error_on_system_not_first() {
        let msgs = vec![
            Message::user("Q"),
            Message::system("late system"),
        ];
        let err = render_chat(&msgs, &ChatRenderOptions::default()).unwrap_err();
        assert_eq!(err, "System message must be at the beginning.");
    }

    #[test]
    fn split_reasoning_matches_jinja_semantics() {
        // Standard case
        let (r, c) = split_reasoning("<think>\nstep one\n</think>\n\nfinal");
        assert_eq!(r, "step one");
        assert_eq!(c, "final");

        // No think block at all
        let (r, c) = split_reasoning("just an answer");
        assert_eq!(r, "");
        assert_eq!(c, "just an answer");

        // Reasoning without opening tag (model omitted <think>)
        let (r, c) = split_reasoning("loose thoughts\n</think>\nanswer");
        assert_eq!(r, "loose thoughts");
        assert_eq!(c, "answer");

        // Multiple close tags: reasoning from first, content from last
        let (r, c) = split_reasoning("<think>a</think><think>b</think>out");
        assert_eq!(r, "a");
        assert_eq!(c, "out");
    }
}
