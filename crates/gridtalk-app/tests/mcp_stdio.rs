//! `gridtalk mcp` as a real process: the handshake, a workbook session
//! and error shapes over stdin/stdout, and how the process exits.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// How long to wait for a frame before failing instead of hanging.
const FRAME_TIMEOUT: Duration = Duration::from_secs(60);

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .canonicalize()
        .unwrap()
}

/// `gridtalk mcp` with these arguments, isolated from the user's
/// config.
fn gridtalk_mcp(args: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_gridtalk"));
    let config = std::env::temp_dir().join(format!("gt-mcp-stdio-cfg-{}", std::process::id()));
    command
        .arg("mcp")
        .args(args)
        .env("GRIDTALK_CONFIG_DIR", &config)
        .env_remove("GRIDTALK_CONFIG");
    command
}

/// A running server and the frames it has written. Threads drain
/// stdout and stderr, so a quiet server fails a read instead of
/// hanging it, and a chatty stderr never fills its pipe. Dropping it
/// (a failed test included) kills a server that is still running.
struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Receiver<String>,
    stderr: Option<JoinHandle<String>>,
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Ok(None) = self.child.try_wait() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

impl Server {
    fn start(args: &[&str]) -> Server {
        let mut child = gridtalk_mcp(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take();
        let mut out = BufReader::new(child.stdout.take().unwrap());
        let (tx, stdout) = mpsc::channel();
        // Lines are decoded lossily, so a frame that is not UTF-8
        // fails the JSON check with its text instead of ending the
        // reader as if stdout had closed.
        thread::spawn(move || {
            let mut line = Vec::new();
            while matches!(out.read_until(b'\n', &mut line), Ok(n) if n > 0) {
                let text = String::from_utf8_lossy(&line);
                let text = text.trim_end_matches(['\n', '\r']).to_string();
                if tx.send(text).is_err() {
                    break;
                }
                line.clear();
            }
        });
        let mut err = child.stderr.take().unwrap();
        let stderr = thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = err.read_to_end(&mut bytes);
            String::from_utf8_lossy(&bytes).into_owned()
        });
        Server {
            child,
            stdin,
            stdout,
            stderr: Some(stderr),
        }
    }

    fn send_line(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().unwrap();
        writeln!(stdin, "{line}").unwrap();
        stdin.flush().unwrap();
    }

    /// The next frame on stdout, which must be one JSON object on one
    /// line.
    fn read(&mut self) -> Value {
        let line = match self.stdout.recv_timeout(FRAME_TIMEOUT) {
            Ok(line) => line,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                panic!("no frame within {FRAME_TIMEOUT:?}")
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => panic!("the server closed stdout"),
        };
        serde_json::from_str(&line).unwrap_or_else(|e| panic!("not a JSON frame ({e}): {line}"))
    }

    /// Send a request and return its response.
    fn request(&mut self, request: Value) -> Value {
        self.send_line(&request.to_string());
        let response = self.read();
        assert_eq!(response["jsonrpc"], "2.0", "{response}");
        assert_eq!(response["id"], request["id"], "{response}");
        response
    }

    fn call(&mut self, id: u64, name: &str, arguments: Value) -> Value {
        self.request(json!({"jsonrpc": "2.0", "id": id, "method": "tools/call", "params": {"name": name, "arguments": arguments}}))
    }

    /// Close stdin and wait: what else the server wrote, how it
    /// exited and its stderr.
    fn finish(mut self) -> (Vec<String>, ExitStatus, String) {
        drop(self.stdin.take());
        let mut rest = Vec::new();
        loop {
            match self.stdout.recv_timeout(FRAME_TIMEOUT) {
                Ok(line) if line.trim().is_empty() => {}
                Ok(line) => rest.push(line),
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    panic!("the server did not exit within {FRAME_TIMEOUT:?} of end of input");
                }
            }
        }
        let status = self.child.wait().unwrap();
        let stderr = self.stderr.take().unwrap().join().unwrap();
        (rest, status, stderr)
    }
}

/// The `{"error": {"kind", ..}}` of a tool result that failed.
fn tool_error_kind(response: &Value) -> &Value {
    assert_eq!(response["result"]["isError"], true, "{response}");
    &response["result"]["structuredContent"]["error"]["kind"]
}

#[test]
fn a_session_over_stdio() {
    let fixtures = fixtures();
    let root = fixtures.to_str().unwrap();
    let mut server = Server::start(&["--root", root, "--timeout", "60"]);

    // Calls before the handshake are refused.
    let early = server.request(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}));
    assert_eq!(early["error"]["code"], -32002, "{early}");

    let init = server.request(json!({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}}));
    assert_eq!(init["result"]["protocolVersion"], "2025-03-26", "{init}");
    assert_eq!(init["result"]["serverInfo"]["name"], "gridtalk");
    // A notification gets no response: the next frame answers ping.
    server.send_line(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
    let ping = server.request(json!({"jsonrpc": "2.0", "id": 3, "method": "ping"}));
    assert_eq!(ping["result"], json!({}));

    let list = server.request(json!({"jsonrpc": "2.0", "id": 4, "method": "tools/list"}));
    let names: Vec<&str> = list["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    for tool in [
        "open_workbook",
        "close_workbook",
        "search",
        "get_cells",
        "what_if",
    ] {
        assert!(names.contains(&tool), "{tool} missing from {names:?}");
    }

    // A workbook session.
    let demo = fixtures.join("demo.csv");
    let open = server.call(5, "open_workbook", json!({"path": demo}));
    assert_eq!(open["result"]["isError"], false, "{open}");
    let id = open["result"]["structuredContent"]["workbook_id"].clone();
    assert!(id.as_str().is_some_and(|s| s.starts_with("wb-")), "{open}");
    let cells = server.call(
        6,
        "get_cells",
        json!({"workbook_id": id, "range": "Sheet1!D2"}),
    );
    assert_eq!(cells["result"]["isError"], false, "{cells}");
    // D2 is =B2*C2 = 2 * 3, evaluated by the server.
    assert_eq!(
        cells["result"]["structuredContent"]["rows"][0][0]["result"]["value"], 6.0,
        "{cells}"
    );
    let reopen = server.call(7, "open_workbook", json!({"path": demo}));
    assert_eq!(reopen["result"]["structuredContent"]["workbook_id"], id);
    assert_eq!(reopen["result"]["structuredContent"]["reused"], true);

    // Error shapes. A tool failure is an `isError` result with a kind.
    let unknown = server.call(
        8,
        "get_cells",
        json!({"workbook_id": "wb-nope", "range": "A1"}),
    );
    assert_eq!(tool_error_kind(&unknown), "unknown_workbook");
    let outside =
        std::env::temp_dir().join(format!("gt-mcp-stdio-outside-{}.csv", std::process::id()));
    std::fs::write(&outside, "A\n1\n").unwrap();
    let denied = server.call(9, "open_workbook", json!({"path": outside}));
    assert_eq!(tool_error_kind(&denied), "path_not_allowed");
    let _ = std::fs::remove_file(&outside);
    // A protocol error is a JSON-RPC error, with the kind in `data`.
    let no_tool = server.call(10, "format_disk", json!({}));
    assert_eq!(no_tool["error"]["code"], -32602, "{no_tool}");
    assert_eq!(no_tool["error"]["data"]["kind"], "unknown_tool");
    let bad_arg = server.call(
        11,
        "get_cells",
        json!({"workbook_id": id, "range": "A1", "extra": 1}),
    );
    assert_eq!(bad_arg["error"]["code"], -32602, "{bad_arg}");
    let no_method = server.request(json!({"jsonrpc": "2.0", "id": 12, "method": "resources/list"}));
    assert_eq!(no_method["error"]["code"], -32601, "{no_method}");
    // A frame that is not JSON is answered with id null.
    server.send_line("{not json");
    let garbled = server.read();
    assert_eq!(garbled["error"]["code"], -32700, "{garbled}");
    assert_eq!(garbled["id"], Value::Null);

    let closed = server.call(13, "close_workbook", json!({"workbook_id": id}));
    assert_eq!(closed["result"]["structuredContent"]["closed"], id);

    // End of input ends the server cleanly, with nothing left unsaid
    // on stdout and its log on stderr.
    let (rest, status, stderr) = server.finish();
    assert!(rest.is_empty(), "unexpected frames: {rest:?}");
    assert!(status.success(), "{status:?}");
    assert!(stderr.contains("serving on stdio"), "{stderr}");
}

#[test]
fn a_request_without_a_newline_before_eof_is_answered() {
    let mut server = Server::start(&[]);
    let stdin = server.stdin.as_mut().unwrap();
    write!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2025-03-26"}}}}"#
    )
    .unwrap();
    let (frames, status, _) = server.finish();
    assert_eq!(frames.len(), 1, "{frames:?}");
    let init: Value = serde_json::from_str(&frames[0]).unwrap();
    assert_eq!(init["id"], 1);
    assert!(init["result"]["protocolVersion"].is_string(), "{init}");
    assert!(status.success(), "{status:?}");
}

#[test]
fn bad_arguments_exit_with_an_error_and_no_frames() {
    let missing = std::env::temp_dir().join("gt-mcp-stdio-no-such-dir");
    let _ = std::fs::remove_dir_all(&missing);
    for args in [
        vec!["--bogus"],
        vec!["--timeout", "0"],
        vec!["--timeout"],
        vec!["--memory-budget", "lots"],
        vec!["--root", missing.to_str().unwrap()],
    ] {
        let output = gridtalk_mcp(&args).stdin(Stdio::null()).output().unwrap();
        assert_eq!(output.status.code(), Some(1), "{args:?}");
        assert!(output.stdout.is_empty(), "{args:?} wrote to stdout");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.starts_with("gridtalk: "), "{args:?}: {stderr}");
    }
}
