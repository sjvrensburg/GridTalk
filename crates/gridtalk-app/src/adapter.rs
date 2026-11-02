//! Phase D (agentic plan): GridTalk adapters — versioned,
//! human-readable harness configuration that the optimise loop
//! evolves without touching model weights.
//!
//! Layout under the adapters root (`<config>/gridtalk/adapters`):
//!
//! ```text
//! <name>/
//!   ADAPTER.md     learned rules for the model (plain markdown)
//!   adapter.toml   knobs: version, top_k, max_steps
//!   CARD.md        eval record, appended on every optimise round
//!   skills/*.md    optional domain skills with `Triggers:` lines
//!   v1/ v2/ …      committed versions (same files inside)
//! ```
//!
//! `adapter.toml`'s `version` is the active pointer; rollback is
//! moving it back. Loading order at investigate time: built-in
//! prompt → adapter rules → matched skills → per-workbook sidecar
//! (`<workbook>.gridtalk.md`, edited next to the spreadsheet).

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct Skill {
    pub name: String,
    /// Lowercase trigger substrings; empty means "always apply".
    pub triggers: Vec<String>,
    pub body: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Adapter {
    pub name: String,
    pub version: u32,
    pub rules: String,
    /// Hybrid-search hits fed to the planner (default 3).
    pub top_k: usize,
    /// Default investigate step budget (default 8).
    pub max_steps: usize,
    /// When a question names a cell (Sheet1!D2), include that
    /// cell's formulas/cells in the context even if retrieval
    /// missed it. Off by default; the optimiser can learn it.
    pub ref_context: bool,
    pub skills: Vec<Skill>,
}

impl Adapter {
    #[allow(dead_code)] // used by tests and future adapter creation UI
    pub fn empty(name: &str) -> Adapter {
        Adapter {
            name: name.into(),
            version: 0,
            rules: String::new(),
            top_k: 3,
            max_steps: 8,
            ref_context: false,
            skills: Vec::new(),
        }
    }

    /// Adapter text appended to the built-in system prompt:
    /// rules, plus skills whose triggers match the question.
    pub fn prompt_for(&self, question: &str) -> String {
        let mut out = String::new();
        if !self.rules.trim().is_empty() {
            out.push_str("\n\nAdapter rules (learned for this workbook shape):\n");
            out.push_str(self.rules.trim());
        }
        let q = question.to_lowercase();
        for skill in &self.skills {
            let applies =
                skill.triggers.is_empty() || skill.triggers.iter().any(|t| q.contains(t.as_str()));
            if applies && !skill.body.trim().is_empty() {
                out.push_str(&format!("\n\nSkill: {}\n{}", skill.name, skill.body.trim()));
            }
        }
        out
    }
}

/// Root holding all adapter directories.
pub fn adapters_root() -> PathBuf {
    if let Ok(dir) = std::env::var("GRIDTALK_CONFIG_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir).join("adapters");
        }
    }
    if let Ok(cfg) = std::env::var("GRIDTALK_CONFIG") {
        if let Some(parent) = PathBuf::from(cfg).parent() {
            return parent.join("adapters");
        }
    }
    if let Some(d) = dirs_path() {
        return d;
    }
    PathBuf::from(".config/gridtalk/adapters")
}

fn dirs_path() -> Option<PathBuf> {
    if let Ok(appdata) = std::env::var("LOCALAPPDATA") {
        if !appdata.is_empty() {
            return Some(PathBuf::from(appdata).join("gridtalk/adapters"));
        }
    }
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        if !xdg.is_empty() {
            return Some(PathBuf::from(xdg).join("gridtalk/adapters"));
        }
    }
    std::env::var("HOME")
        .ok()
        .filter(|h| !h.is_empty())
        .map(|h| PathBuf::from(h).join(".config/gridtalk/adapters"))
}

/// Adapter names are a single path component. Anything else
/// (separators, `..`, empty) is rejected so a name from the CLI,
/// prefs or a tasks file can never escape the adapters root.
pub fn valid_adapter_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains('/')
        && !name.contains('\\')
        && !name.contains('\0')
}

pub fn adapter_dir(root: &Path, name: &str) -> PathBuf {
    if valid_adapter_name(name) {
        root.join(name)
    } else {
        // Never escape the root; callers that care validate first.
        root.join("__invalid__")
    }
}

fn clean_value(raw: &str) -> String {
    let v = raw.trim();
    if let Some(stripped) = v.strip_prefix('"') {
        // Quoted: take up to the closing quote.
        return stripped.split('"').next().unwrap_or("").to_string();
    }
    // Unquoted: strip a trailing comment.
    v.split(" #").next().unwrap_or(v).trim().to_string()
}

fn parse_knobs(text: &str) -> (u32, usize, usize, bool) {
    let (mut version, mut top_k, mut max_steps) = (0u32, 3usize, 8usize);
    let mut ref_context = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.starts_with('[') {
            continue;
        }
        if let Some((key, raw)) = line.split_once('=') {
            let value = clean_value(raw);
            let warn = |key: &str| {
                eprintln!("gridtalk: warning: malformed adapter knob {key}={raw:?}; using default");
            };
            match key.trim() {
                "version" => match value.parse() {
                    Ok(v) => version = v,
                    Err(_) => warn("version"),
                },
                "top_k" => match value.parse::<usize>() {
                    Ok(v) => top_k = v.clamp(1, 25),
                    Err(_) => warn("top_k"),
                },
                "max_steps" => match value.parse::<usize>() {
                    Ok(v) => max_steps = v.clamp(1, 64),
                    Err(_) => warn("max_steps"),
                },
                "ref_context" => ref_context = value == "true",
                _ => {}
            }
        }
    }
    (version, top_k, max_steps, ref_context)
}

fn knob_text(a: &Adapter) -> String {
    format!(
        "# GridTalk adapter knobs for {}\nversion = {}\ntop_k = {}\nmax_steps = {}\nref_context = {}\n",
        a.name, a.version, a.top_k, a.max_steps, a.ref_context
    )
}

fn load_skills(dir: &Path) -> Vec<Skill> {
    let mut skills = Vec::new();
    let skills_dir = dir.join("skills");
    if let Ok(entries) = std::fs::read_dir(&skills_dir) {
        let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();
        for path in paths {
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let name = path
                .file_stem()
                .and_then(|n| n.to_str())
                .unwrap_or("skill")
                .to_string();
            let mut triggers = Vec::new();
            let mut body_lines = Vec::new();
            // Only a Triggers line in the first three lines is
            // metadata; later ones are body, verbatim.
            for (idx, line) in text.lines().enumerate() {
                let header = idx < 3;
                let parsed = if header {
                    line.trim()
                        .strip_prefix("Triggers:")
                        .or_else(|| line.trim().strip_prefix("triggers:"))
                } else {
                    None
                };
                if let Some(rest) = parsed {
                    let parsed: Vec<String> = rest
                        .split(',')
                        .map(|t| t.trim().to_lowercase())
                        .filter(|t| !t.is_empty())
                        .collect();
                    // An explicit empty Triggers line means
                    // "never auto-apply", not "always".
                    triggers = if parsed.is_empty() {
                        vec!["\u{1}never".to_string()]
                    } else {
                        parsed
                    };
                } else {
                    body_lines.push(line);
                }
            }
            skills.push(Skill {
                name,
                triggers,
                body: body_lines.join("\n"),
            });
        }
    }
    skills
}

fn load_from_dir(
    dir: &Path,
    name: &str,
    version: u32,
    top_k: usize,
    max_steps: usize,
    ref_context: bool,
) -> Adapter {
    // Per-version knobs win over the root pointer file (the root
    // file holds only the version pointer plus a convenience copy).
    let (top_k, max_steps, ref_context) = match std::fs::read_to_string(dir.join("adapter.toml")) {
        Ok(text) if !text.is_empty() => {
            let (_, tk, ms, rc) = parse_knobs(&text);
            (tk, ms, rc)
        }
        _ => (top_k, max_steps, ref_context),
    };
    let rules = std::fs::read_to_string(dir.join("ADAPTER.md")).unwrap_or_default();
    Adapter {
        name: name.into(),
        version,
        rules,
        top_k,
        max_steps,
        ref_context,
        skills: load_skills(dir),
    }
}

/// Load the active version of an adapter. A missing adapter is an
/// empty adapter (version 0), not an error — chat works unadapted.
pub fn load_adapter(root: &Path, name: &str) -> Adapter {
    let dir = adapter_dir(root, name);
    let knobs_text = std::fs::read_to_string(dir.join("adapter.toml")).unwrap_or_default();
    let (version, top_k, max_steps, ref_context) = parse_knobs(&knobs_text);
    if version > 0 {
        let vdir = dir.join(format!("v{version}"));
        if vdir.is_dir() {
            return load_from_dir(&vdir, name, version, top_k, max_steps, ref_context);
        }
        // Pointer names a version that is not on disk: report v0
        // rather than claiming a version we did not load.
        eprintln!(
            "gridtalk: warning: adapter {name} points at missing v{version}; loading root copies as v0"
        );
        return load_from_dir(&dir, name, 0, top_k, max_steps, ref_context);
    }
    load_from_dir(&dir, name, version, top_k, max_steps, ref_context)
}

pub fn list_adapters(root: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(root)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| e.path().is_dir())
                .filter_map(|e| e.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

fn scopeguard_remove(path: &Path) -> impl FnOnce() {
    let owned = path.to_path_buf();
    move || {
        let _ = owned;
    }
}

/// Commit `adapter` as the next version: writes `vN/` and moves
/// the root pointer. Returns the committed version.
pub fn commit_version(root: &Path, adapter: &Adapter) -> Result<u32, String> {
    if !valid_adapter_name(&adapter.name) {
        return Err(format!("invalid adapter name {:?}", adapter.name));
    }
    let dir = adapter_dir(root, &adapter.name);
    // Next version is the max on disk too — a stale in-memory
    // adapter can never rewrite an existing version.
    let on_disk_max = std::fs::read_dir(&dir)
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|e| {
                    e.file_name()
                        .to_str()
                        .and_then(|n| n.strip_prefix('v'))
                        .and_then(|n| n.parse::<u32>().ok())
                })
                .max()
                .unwrap_or(0)
        })
        .unwrap_or(0);
    let next = adapter.version.max(on_disk_max) + 1;
    let vdir = dir.join(format!("v{next}"));
    if vdir.exists() {
        return Err(format!("adapter version v{next} already exists"));
    }
    // Write into a temp dir and rename into place: no partial vN.
    let tmp = dir.join(format!(".v{next}.tmp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(tmp.join("skills")).map_err(|e| e.to_string())?;
    let vdir_tmp = tmp.clone();
    let _guard = scopeguard_remove(&tmp);
    let mut committed = adapter.clone();
    committed.version = next;
    let write_result = (|| -> Result<(), String> {
        std::fs::write(vdir_tmp.join("ADAPTER.md"), &committed.rules).map_err(|e| e.to_string())?;
        std::fs::write(vdir_tmp.join("adapter.toml"), knob_text(&committed))
            .map_err(|e| e.to_string())?;
        for skill in &committed.skills {
            let mut text = String::new();
            if !skill.triggers.is_empty() {
                text.push_str(&format!("Triggers: {}\n", skill.triggers.join(", ")));
            }
            text.push_str(&skill.body);
            std::fs::write(
                vdir_tmp.join("skills").join(format!("{}.md", skill.name)),
                text,
            )
            .map_err(|e| e.to_string())?;
        }
        std::fs::rename(&vdir_tmp, &vdir).map_err(|e| e.to_string())
    })();
    if let Err(e) = write_result {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(e);
    }
    // Root pointer + a copy of the active files for easy reading.
    std::fs::write(dir.join("adapter.toml"), knob_text(&committed)).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("ADAPTER.md"), &committed.rules).map_err(|e| e.to_string())?;
    Ok(next)
}

/// Move the active pointer back one version (no history rewrite).
pub fn rollback(root: &Path, name: &str) -> Result<u32, String> {
    let current = load_adapter(root, name);
    if current.version == 0 {
        return Err(format!("adapter {name} has no committed versions"));
    }
    let dir = adapter_dir(root, name);
    let target = current.version - 1;
    let mut pointer = current.clone();
    pointer.version = target;
    std::fs::write(dir.join("adapter.toml"), knob_text(&pointer)).map_err(|e| e.to_string())?;
    // Refresh the root convenience copy from the rolled-back
    // version (v0 clears it — no resurrected rules; note the root
    // copy itself must not be the source for v0).
    let root_rules = if target == 0 {
        String::new()
    } else {
        load_adapter(root, name).rules
    };
    std::fs::write(dir.join("ADAPTER.md"), &root_rules).map_err(|e| e.to_string())?;
    Ok(target)
}

/// Append an eval record to the adapter's CARD.md.
pub fn append_card(root: &Path, name: &str, markdown: &str) -> Result<(), String> {
    let dir = adapter_dir(root, name);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    use std::io::Write as _;
    let path = dir.join("CARD.md");
    let exists = path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| e.to_string())?;
    if !exists {
        file.write_all(b"# Adapter CARD\n")
            .map_err(|e| e.to_string())?;
    }
    file.write_all(markdown.as_bytes())
        .map_err(|e| e.to_string())
}

/// Per-workbook sidecar rules (`<workbook>.gridtalk.md`), the
/// AGENT.md-style file a user edits next to the spreadsheet.
pub fn sidecar_rules(workbook_path: &Path) -> Option<String> {
    let sidecar = PathBuf::from(format!("{}.gridtalk.md", workbook_path.display()));
    std::fs::read_to_string(sidecar)
        .ok()
        .filter(|t| !t.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("gt-adapter-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn commit_load_and_rollback_versions() {
        let root = temp_root("versions");
        let mut a = Adapter::empty("demo");
        a.rules = "- Answer with just the final number.\n".into();
        a.top_k = 5;
        let v = commit_version(&root, &a).unwrap();
        assert_eq!(v, 1);
        let loaded = load_adapter(&root, "demo");
        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.top_k, 5);
        assert!(loaded.rules.contains("final number"));

        let mut a2 = loaded.clone();
        a2.rules.push_str("- Cite the range.\n");
        assert_eq!(commit_version(&root, &a2).unwrap(), 2);
        assert_eq!(load_adapter(&root, "demo").version, 2);
        assert_eq!(rollback(&root, "demo").unwrap(), 1);
        let back = load_adapter(&root, "demo");
        assert_eq!(back.version, 1);
        assert!(!back.rules.contains("Cite the range"));
        // Rolling back to v0 clears the rules entirely.
        assert_eq!(rollback(&root, "demo").unwrap(), 0);
        assert!(load_adapter(&root, "demo").rules.trim().is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn skills_match_triggers_and_prompt_order() {
        let root = temp_root("skills");
        let mut a = Adapter::empty("demo");
        a.rules = "- Be terse.".into();
        a.skills = vec![
            Skill {
                name: "provisions".into(),
                triggers: vec!["provision".into()],
                body: "Check the Provision row.".into(),
            },
            Skill {
                name: "always".into(),
                triggers: vec![],
                body: "Always cite.".into(),
            },
        ];
        commit_version(&root, &a).unwrap();
        let loaded = load_adapter(&root, "demo");
        let p = loaded.prompt_for("what is the provision?");
        assert!(p.contains("Be terse."));
        assert!(p.contains("Check the Provision row."));
        assert!(p.contains("Always cite."));
        let p2 = loaded.prompt_for("total qty?");
        assert!(!p2.contains("Check the Provision row."));
        let _ = std::fs::remove_dir_all(&root);
    }
}
