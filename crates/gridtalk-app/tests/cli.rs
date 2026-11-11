//! The `gridtalk` binary's help, version, strict argument parsing and
//! exit codes (0 ok, 1 runtime, 2 usage, 3 input, 4 backend, 5 tool).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn demo() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/demo.csv")
        .to_str()
        .unwrap()
        .to_string()
}

/// A config directory with no config, so the local echo backend runs.
fn no_config() -> PathBuf {
    std::env::temp_dir().join(format!("gt-cli-cfg-{}", std::process::id()))
}

fn gridtalk(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gridtalk"))
        .args(args)
        .env("GRIDTALK_CONFIG_DIR", no_config())
        .env_remove("GRIDTALK_CONFIG")
        .output()
        .unwrap()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// Run and require exit `code`; returns the output.
fn expect(args: &[&str], code: i32) -> Output {
    let output = gridtalk(args);
    assert_eq!(
        output.status.code(),
        Some(code),
        "{args:?}\nstdout: {}\nstderr: {}",
        stdout(&output),
        stderr(&output)
    );
    output
}

#[test]
fn version_and_help() {
    let want = format!("gridtalk {}\n", env!("CARGO_PKG_VERSION"));
    for flag in ["--version", "-V", "version"] {
        assert_eq!(stdout(&expect(&[flag], 0)), want);
    }
    expect(&["--version", "extra"], 2);
    assert!(stdout(&expect(&["version", "--help"], 0)).contains("Print the version."));
    assert!(stdout(&expect(&["help", "--help"], 0)).contains("Exit codes: 0 success"));

    for args in [&[][..], &["help"], &["--help"], &["-h"]] {
        let top = stdout(&expect(args, 0));
        for usage in [
            "gridtalk ask <file> <question...>",
            "gridtalk bench run <file>",
            "gridtalk mcp [--root <dir>]...",
            "Exit codes: 0 success",
        ] {
            assert!(top.contains(usage), "{args:?} lacks {usage}:\n{top}");
        }
    }
    // One command's help, three ways, and through a subcommand.
    for args in [
        &["help", "bench"][..],
        &["bench", "--help"],
        &["bench", "-h"],
        &["bench", "run", "--help"],
    ] {
        let help = stdout(&expect(args, 0));
        assert!(
            help.contains("gridtalk bench generate <file>"),
            "{args:?}: {help}"
        );
        assert!(
            help.contains("Generate evaluation tasks"),
            "{args:?}: {help}"
        );
        assert!(!help.contains("gridtalk ask"), "{args:?}: {help}");
    }
    expect(&["help", "nope"], 2);
    // The American spelling is an alias.
    assert!(stdout(&expect(&["help", "optimize"], 0)).contains("gridtalk optimise <file>"));
    expect(&["optimize", "--rounds", "1"], 2);
}

#[test]
fn usage_errors_exit_2_and_name_the_problem() {
    let demo = demo();
    let missing = "gt-cli-no-such-file.csv";
    for (args, problem) in [
        (vec!["frobnicate"], "unknown command"),
        (vec!["graph"], "missing <file>"),
        (vec!["graph", &demo, "extra"], "unexpected argument"),
        // A flag is never taken for the file.
        (
            vec!["bench", "generate", "--out", "x.json"],
            "missing <file>",
        ),
        (
            vec!["bench", "generate", &demo, "--tasks", "t.json"],
            "unknown flag --tasks",
        ),
        (
            vec!["bench", "generate", &demo, "--max", "0"],
            "--max must be",
        ),
        (
            vec!["bench", "run", missing, "--split", "nope"],
            "unknown split",
        ),
        (
            vec!["bench", "run", &demo, "--card"],
            "--card needs a value",
        ),
        (
            vec!["bench", "run", &demo, "--card", "a", "--card", "b"],
            "given twice",
        ),
        (vec!["bench"], "missing subcommand"),
        (
            vec!["optimise", &demo, "--rounds", "11"],
            "--rounds must be 1 to 10",
        ),
        (vec!["adapter", "show", "a", "b"], "unexpected argument"),
        (vec!["adapter", "frob"], "unknown subcommand"),
        (
            vec!["models", "import", "a.gguf", "b"],
            "unexpected argument",
        ),
        (vec!["models", "download"], "missing <url>"),
        (vec!["tools", "call", &demo], "missing '<json>'"),
        (
            vec!["tools", "call", missing, "{not json"],
            "invalid tool request JSON",
        ),
        (vec!["tools", "list", "extra"], "unexpected argument"),
        (vec!["ask", &demo], "missing the question"),
        (
            vec!["ask", "--adapter", "x", "question"],
            "unknown flag --adapter",
        ),
        // Flags may come before the file; the first word after it
        // starts the question.
        (
            vec!["investigate", "--adapter", "x", "question"],
            "missing the question",
        ),
        (
            vec!["investigate", &demo, "--adaptor", "fin", "total?"],
            "unknown flag --adaptor",
        ),
        (
            vec!["investigate", &demo, "--adapter", "../x", "total?"],
            "invalid adapter name",
        ),
        (
            vec![
                "investigate",
                &demo,
                "--adapter",
                "a",
                "--adapter",
                "b",
                "q",
            ],
            "--adapter given twice",
        ),
        (
            vec!["mcp", "--allow-write", "--allow-write"],
            "--allow-write given twice",
        ),
        (vec!["mcp", "stray"], "unexpected argument"),
        (vec!["mcp", "--root"], "--root needs a value"),
        (
            vec!["investigate", &demo, "--adapter"],
            "--adapter needs a value",
        ),
        (vec!["generate", "model.gguf"], "missing the prompt"),
        (vec!["mcp", "--timeout", "0"], "--timeout must be"),
        (vec!["graph", "-o", "x"], "unknown flag -o"),
        (
            vec!["ask", &demo, "--adapter", "x", "q"],
            "unknown flag --adapter",
        ),
        (
            vec!["investigate", &demo, "--adaptor", "x", "q"],
            "unknown flag --adaptor",
        ),
        (
            vec!["optimise", &demo, "--adapter", "../x"],
            "invalid adapter name",
        ),
        (vec!["adapter", "rollback", "../x"], "invalid adapter name"),
        (vec!["adapter", "export", "../x"], "invalid adapter name"),
    ] {
        let output = expect(&args, 2);
        let err = stderr(&output);
        assert!(err.starts_with("gridtalk: "), "{args:?}: {err}");
        assert!(
            err.contains(problem),
            "{args:?} should say {problem:?}: {err}"
        );
        assert!(stdout(&output).is_empty(), "{args:?} wrote to stdout");
        if problem != "unknown command" {
            assert!(err.contains("usage: gridtalk"), "{args:?}: {err}");
        }
    }
}

#[test]
fn input_backend_and_tool_failures_have_their_own_codes() {
    let demo = demo();
    let missing = "gt-cli-no-such-file.csv";
    // 3: an input that cannot be read.
    let err = stderr(&expect(&["graph", missing], 3));
    assert!(err.contains(missing), "{err}");
    expect(
        &["tools", "call", missing, r#"{"tool":"workbook_info"}"#],
        3,
    );
    expect(
        &[
            "bench",
            "run",
            &demo,
            "--tasks",
            "gt-cli-no-such-tasks.json",
        ],
        3,
    );
    expect(&["spike", &demo], 3);
    expect(&["generate", "gt-cli-no-such-model.gguf", "hi"], 3);
    // 5: the tool ran and reported an error, printed once on stdout.
    let failed = expect(
        &[
            "tools",
            "call",
            &demo,
            r#"{"tool":"get_cells","range":"Nope!A1"}"#,
        ],
        5,
    );
    let result: serde_json::Value = serde_json::from_str(&stdout(&failed)).unwrap();
    assert_eq!(result["ok"], false, "{result}");
    assert!(stderr(&failed).is_empty(), "{}", stderr(&failed));
    // 4: a backend that cannot be reached (nothing listens on port 9).
    let dir = no_config().with_extension("backend");
    std::fs::create_dir_all(&dir).unwrap();
    let config = dir.join("config.toml");
    std::fs::write(
        &config,
        "[inference]\nbackend = \"openai-compatible\"\nbase_url = \"http://127.0.0.1:9/v1\"\nmodel = \"m\"\n",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_gridtalk"))
        .args(["ask", &demo, "apples"])
        .env("GRIDTALK_CONFIG", &config)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(4), "{}", stderr(&output));
    // optimise scores through the same backend (its adapters live
    // beside the config, in this scratch directory).
    let output = Command::new(env!("CARGO_BIN_EXE_gridtalk"))
        .args(["optimise", &demo, "--rounds", "1"])
        .env("GRIDTALK_CONFIG", &config)
        .env_remove("GRIDTALK_CONFIG_DIR")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(4), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("scoring failed"),
        "{}",
        stderr(&output)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn commands_succeed_with_exit_0() {
    let demo = demo();
    let graph = stdout(&expect(&["graph", &demo], 0));
    assert!(graph.contains("[region]"), "{graph}");
    let answer = stdout(&expect(&["ask", &demo, "how", "many", "apples"], 0));
    assert!(answer.contains("Sheet1!"), "{answer}");
    let cells = stdout(&expect(
        &[
            "tools",
            "call",
            &demo,
            r#"{"tool":"get_cells","range":"Sheet1!A1:B2"}"#,
        ],
        0,
    ));
    assert!(cells.contains("\"ok\": true"), "{cells}");
}
