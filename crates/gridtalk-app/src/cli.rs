//! Command-line plumbing shared by every `gridtalk` command: the
//! command table behind `help`, typed errors with distinct exit codes,
//! and a strict flag parser.

use std::collections::BTreeMap;
use std::io::Write;

/// Why a command failed, which decides the process exit code.
#[derive(Debug, PartialEq)]
pub enum CliError {
    /// Bad arguments: unknown command or flag, a missing or extra
    /// argument, a value out of range (exit 2).
    Usage(String),
    /// A workbook, model, tasks file or adapter could not be read or
    /// is invalid (exit 3).
    Input(String),
    /// The inference backend failed (exit 4).
    Backend(String),
    /// `tools call` ran and the tool reported an error; its result is
    /// already printed (exit 5).
    ToolFailed,
    /// Anything else (exit 1).
    Other(String),
}

impl CliError {
    pub fn exit_code(&self) -> i32 {
        match self {
            CliError::Other(_) => 1,
            CliError::Usage(_) => 2,
            CliError::Input(_) => 3,
            CliError::Backend(_) => 4,
            CliError::ToolFailed => 5,
        }
    }

    /// The message for stderr (`None` when the output already says it).
    pub fn message(&self) -> Option<&str> {
        match self {
            CliError::Usage(m) | CliError::Input(m) | CliError::Backend(m) | CliError::Other(m) => {
                Some(m)
            }
            CliError::ToolFailed => None,
        }
    }
}

impl From<String> for CliError {
    fn from(message: String) -> Self {
        CliError::Other(message)
    }
}

impl From<&str> for CliError {
    fn from(message: &str) -> Self {
        CliError::Other(message.to_string())
    }
}

/// The exit codes, for the top-level help.
pub const EXIT_CODES: &str = "Exit codes: 0 success, 1 runtime failure, 2 usage error, 3 input could not be read (workbook, model, tasks file or adapter), 4 inference backend failed, 5 the tool call reported an error";

/// One command: its name, usage lines and a one-line summary.
pub struct Command {
    pub name: &'static str,
    pub usage: &'static [&'static str],
    pub summary: &'static str,
}

/// Every command, in the order `gridtalk help` lists them.
pub const COMMANDS: &[Command] = &[
    Command {
        name: "ask",
        usage: &["gridtalk ask <file> <question...>"],
        summary: "Find the regions that best match the question and answer from them, citing their ranges.",
    },
    Command {
        name: "investigate",
        usage: &["gridtalk investigate <file> [--adapter <name>] <question...>"],
        summary: "Answer with the tool-using planner (search, get_cells, what_if...), printing each tool step and the citations. --adapter must come before the question.",
    },
    Command {
        name: "graph",
        usage: &["gridtalk graph <file>"],
        summary: "Print the workbook's property graph: its nodes, then its edges.",
    },
    Command {
        name: "view",
        usage: &["gridtalk view <file>"],
        summary: "Print the first sheet's grid and the graph nodes, and show the grid/graph selection sync.",
    },
    Command {
        name: "tools",
        usage: &["gridtalk tools list", "gridtalk tools call <file> '<json>'"],
        summary: "List the tools' JSON schemas, or run one tool request (e.g. '{\"tool\":\"get_cells\",\"range\":\"Sheet1!A1:B3\"}') and print its result; a failed tool exits 5.",
    },
    Command {
        name: "mcp",
        usage: &["gridtalk mcp [--root <dir>]... [--timeout <seconds>] [--memory-budget <MiB>] [--allow-write]"],
        summary: "Serve the tools over MCP on stdio (see docs/mcp.md). --root may repeat; --allow-write is accepted and does nothing.",
    },
    Command {
        name: "bench",
        usage: &[
            "gridtalk bench generate <file> [--out tasks.json] [--max <n>]",
            "gridtalk bench run <file> [--tasks tasks.json] [--split train|dev|all] [--card CARD.md] [--adapter <name>]",
        ],
        summary: "Generate evaluation tasks from a workbook (default 100), or run them against the configured backend and print or write the CARD.",
    },
    Command {
        name: "optimise",
        usage: &["gridtalk optimise <file> [--adapter <name>] [--rounds <1-10>]"],
        summary: "Evolve an adapter's rules on generated tasks, keeping a candidate only when it scores better (default adapter \"default\", 3 rounds). Also spelled optimize.",
    },
    Command {
        name: "adapter",
        usage: &[
            "gridtalk adapter list",
            "gridtalk adapter show <name>",
            "gridtalk adapter rollback <name>",
            "gridtalk adapter export <name> [--out AGENTS.md]",
        ],
        summary: "List, inspect, roll back or export the versioned adapters.",
    },
    Command {
        name: "models",
        usage: &[
            "gridtalk models",
            "gridtalk models import <file.gguf>",
            "gridtalk models download <http(s)://.../file.gguf> [name]",
            "gridtalk models remove <name>",
        ],
        summary: "List, import, download or remove GGUF models in the model cache.",
    },
    Command {
        name: "generate",
        usage: &["gridtalk generate <model.gguf> <prompt...>"],
        summary: "Generate from a GGUF model with llama.cpp and print timing statistics.",
    },
    Command {
        name: "spike",
        usage: &["gridtalk spike <model.gguf>"],
        summary: "Check a GGUF model's header without loading it (exit 3 when it does not validate).",
    },
    Command {
        name: "open-in-gridcraft",
        usage: &["gridtalk open-in-gridcraft <file>"],
        summary: "Launch the Gridcraft editor on a file (GRIDTALK_GRIDCRAFT_BIN overrides where it is found).",
    },
    Command {
        name: "help",
        usage: &["gridtalk help [<command>]", "gridtalk <command> --help"],
        summary: "Print this help, or one command's.",
    },
    Command {
        name: "version",
        usage: &["gridtalk --version"],
        summary: "Print the version.",
    },
];

/// Commands whose first argument is a subcommand, so `gridtalk <cmd>
/// <sub> --help` is help too.
pub const WITH_SUBCOMMANDS: &[&str] = &["adapter", "bench", "models", "tools"];

pub fn command(name: &str) -> Option<&'static Command> {
    COMMANDS.iter().find(|c| c.name == canonical(name))
}

/// A command's own name for an alias (`optimize` is `optimise`).
pub fn canonical(name: &str) -> &str {
    match name {
        "optimize" => "optimise",
        other => other,
    }
}

pub fn version() -> String {
    format!("gridtalk {}", env!("CARGO_PKG_VERSION"))
}

/// `gridtalk help`: every command's usage, then config and exit codes.
pub fn top_help() -> String {
    let mut out = String::from("GridTalk — talk to your spreadsheet, locally.\n\nUsage:\n");
    for c in COMMANDS {
        for line in c.usage {
            out.push_str(&format!("  {line}\n"));
        }
    }
    out.push_str(
        "\nFiles: .csv, .tsv, .xlsx, .xlsm, .xltx, .xltm, .xls, .xlsb (values only) and .ods (values only).\nRun `gridtalk help <command>` for one command.\nConfig: GRIDTALK_CONFIG=/path/to/config.toml (default: <config dir>/gridtalk/config.toml, else the local echo backend).\n",
    );
    out.push_str(EXIT_CODES);
    out
}

/// `gridtalk help <command>`.
pub fn command_help(c: &Command) -> String {
    let mut out = String::from("Usage:\n");
    for line in c.usage {
        out.push_str(&format!("  {line}\n"));
    }
    out.push('\n');
    out.push_str(c.summary);
    out
}

/// A usage error that points at the command's help.
pub fn usage(command: &str, problem: impl std::fmt::Display) -> CliError {
    let lines = self::command(command)
        .map(|c| c.usage.join("\n       "))
        .unwrap_or_default();
    CliError::Usage(format!(
        "{problem}\nusage: {lines}\n(run `gridtalk help {command}` for more)"
    ))
}

pub fn is_help(arg: &str) -> bool {
    arg == "--help" || arg == "-h"
}

/// Arguments split into positionals and `--flag value` pairs.
#[derive(Debug)]
pub struct Parsed {
    pub positional: Vec<String>,
    flags: BTreeMap<&'static str, String>,
}

impl Parsed {
    pub fn flag(&self, name: &str) -> Option<&str> {
        self.flags.get(name).map(String::as_str)
    }

    /// The positionals: one for each of `required` (named in the error
    /// when missing, e.g. `"<file>"`), then up to `optional` more.
    pub fn positionals(
        &self,
        command: &str,
        required: &[&str],
        optional: usize,
    ) -> Result<&[String], CliError> {
        let n = self.positional.len();
        if let Some(missing) = required.get(n) {
            return Err(usage(command, format!("missing {missing}")));
        }
        if let Some(extra) = self.positional.get(required.len() + optional) {
            return Err(usage(command, format!("unexpected argument {extra:?}")));
        }
        Ok(&self.positional)
    }
}

/// Parse `args` strictly: every `--flag` must be one of `flags`, given
/// once, with a value that is not itself a flag. A short option
/// (`-o`) is an unknown flag, not a file name; a lone `-` and anything
/// else is a positional.
pub fn parse(command: &str, args: &[String], flags: &[&'static str]) -> Result<Parsed, CliError> {
    let mut parsed = Parsed {
        positional: Vec::new(),
        flags: BTreeMap::new(),
    };
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        if !arg.starts_with('-') || arg == "-" {
            parsed.positional.push(arg.clone());
            continue;
        }
        let Some(name) = flags.iter().copied().find(|f| f == arg) else {
            return Err(usage(command, format!("unknown flag {arg}")));
        };
        let value = rest
            .next()
            .filter(|v| !v.starts_with("--"))
            .ok_or_else(|| usage(command, format!("{name} needs a value")))?;
        if parsed.flags.insert(name, value.clone()).is_some() {
            return Err(usage(command, format!("{name} given twice")));
        }
    }
    Ok(parsed)
}

/// Prints streamed tokens as they arrive, flushing each so they are
/// seen at once. When stdout is closed (`| head`), it raises `cancel`
/// so generation stops instead of the process panicking.
pub struct TokenSink<'a> {
    out: std::io::StdoutLock<'static>,
    cancel: &'a std::sync::atomic::AtomicBool,
    pub closed: bool,
}

impl<'a> TokenSink<'a> {
    pub fn new(cancel: &'a std::sync::atomic::AtomicBool) -> Self {
        TokenSink {
            out: std::io::stdout().lock(),
            cancel,
            closed: false,
        }
    }

    pub fn write(&mut self, text: &str) {
        if self.closed {
            return;
        }
        if self
            .out
            .write_all(text.as_bytes())
            .and_then(|()| self.out.flush())
            .is_err()
        {
            self.closed = true;
            self.cancel
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn the_parser_is_strict() {
        let p = parse(
            "bench",
            &args(&["book.csv", "--out", "t.json"]),
            &["--out", "--max"],
        )
        .unwrap();
        assert_eq!(p.positional, ["book.csv"]);
        assert_eq!(p.flag("--out"), Some("t.json"));
        assert_eq!(p.flag("--max"), None);
        for (bad, why) in [
            (&["--tasks", "x"][..], "unknown flag --tasks"),
            (&["-o", "x"][..], "unknown flag -o"),
            (&["--out"][..], "--out needs a value"),
            (&["--out", "--max", "3"][..], "--out needs a value"),
            (&["--out", "a", "--out", "b"][..], "--out given twice"),
        ] {
            let err = parse("bench", &args(bad), &["--out", "--max"]).unwrap_err();
            assert_eq!(err.exit_code(), 2);
            assert!(err.message().unwrap().starts_with(why), "{err:?}");
        }
        let p = parse("graph", &args(&["a.csv", "b.csv"]), &[]).unwrap();
        let err = p.positionals("graph", &["<file>"], 0).unwrap_err();
        assert!(
            err.message()
                .unwrap()
                .starts_with("unexpected argument \"b.csv\""),
            "{err:?}"
        );
        let err = p
            .positionals("graph", &["<a>", "<b>", "<c>"], 0)
            .unwrap_err();
        assert!(err.message().unwrap().starts_with("missing <c>"), "{err:?}");
        assert_eq!(p.positionals("graph", &["<file>"], 1).unwrap().len(), 2);
    }

    #[test]
    fn every_command_has_help_and_the_table_is_unique() {
        let mut names: Vec<&str> = COMMANDS.iter().map(|c| c.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), COMMANDS.len());
        let top = top_help();
        for c in COMMANDS {
            assert!(!c.usage.is_empty() && !c.summary.is_empty(), "{}", c.name);
            for line in c.usage {
                assert!(top.contains(line), "{line}");
            }
            assert!(command_help(c).contains(c.summary));
        }
        assert!(top.contains(EXIT_CODES));
        for name in WITH_SUBCOMMANDS {
            assert!(command(name).is_some(), "{name} is not in COMMANDS");
        }
    }
}
