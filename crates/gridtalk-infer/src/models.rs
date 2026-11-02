//! GGUF model management: cache directory, metadata parsing,
//! import/remove, and plain-http download (for local mirrors and
//! future proxies). No model is ever downloaded implicitly.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct GgufInfo {
    pub version: u32,
    pub tensor_count: u64,
    pub architecture: Option<String>,
    pub name: Option<String>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug)]
pub enum ModelError {
    NotGguf(String),
    Truncated(String),
    Io(std::io::Error),
    Http(String),
    InvalidName(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::NotGguf(m) => write!(f, "not a GGUF file: {m}"),
            ModelError::Truncated(m) => write!(f, "truncated GGUF: {m}"),
            ModelError::Io(e) => write!(f, "io error: {e}"),
            ModelError::Http(m) => write!(f, "download error: {m}"),
            ModelError::InvalidName(m) => write!(f, "invalid model name: {m}"),
        }
    }
}
impl std::error::Error for ModelError {}
impl From<std::io::Error> for ModelError {
    fn from(e: std::io::Error) -> Self {
        ModelError::Io(e)
    }
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ModelError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| ModelError::NotGguf("length overflow in GGUF header".into()))?;
        if end > self.b.len() {
            return Err(ModelError::Truncated(format!(
                "need {n} bytes at {} of {}",
                self.pos,
                self.b.len()
            )));
        }
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, ModelError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, ModelError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, ModelError> {
        let len = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(len)?).into_owned())
    }
    fn skip_value(&mut self, ty: u32) -> Result<Option<String>, ModelError> {
        self.skip_value_depth(ty, 0)
    }

    fn skip_value_depth(&mut self, ty: u32, depth: usize) -> Result<Option<String>, ModelError> {
        if depth > 16 {
            return Err(ModelError::NotGguf(
                "GGUF metadata nested deeper than 16".into(),
            ));
        }
        match ty {
            0 | 1 | 7 => {
                self.take(1)?;
                Ok(None)
            }
            2 | 3 => {
                self.take(2)?;
                Ok(None)
            }
            4 => Ok(Some(self.u32()?.to_string())),
            5 => {
                let v = self.u32()? as i32;
                Ok(Some(v.to_string()))
            }
            6 => {
                let v = f32::from_le_bytes(self.take(4)?.try_into().unwrap());
                Ok(Some(v.to_string()))
            }
            8 => Ok(Some(self.string()?)),
            9 => {
                let elem = self.u32()?;
                let count = self.u64()?;
                if count > 1_000_000 {
                    return Err(ModelError::NotGguf(format!(
                        "GGUF array of {count} elements exceeds sanity cap"
                    )));
                }
                for _ in 0..count {
                    self.skip_value_depth(elem, depth + 1)?;
                }
                Ok(None)
            }
            10 => Ok(Some(self.u64()?.to_string())),
            11 => {
                let v = self.u64()? as i64;
                Ok(Some(v.to_string()))
            }
            12 => {
                let v = f64::from_le_bytes(self.take(8)?.try_into().unwrap());
                Ok(Some(v.to_string()))
            }
            other => Err(ModelError::NotGguf(format!(
                "unknown metadata value type {other}"
            ))),
        }
    }
}

/// Parse a GGUF header (v2/v3 layout) without loading tensors.
pub fn parse_gguf(bytes: &[u8]) -> Result<GgufInfo, ModelError> {
    let mut r = Reader { b: bytes, pos: 0 };
    if r.take(4)? != b"GGUF" {
        return Err(ModelError::NotGguf("bad magic".into()));
    }
    let version = r.u32()?;
    if version != 2 && version != 3 {
        return Err(ModelError::NotGguf(format!(
            "unsupported GGUF version {version}"
        )));
    }
    let tensor_count = r.u64()?;
    let kv_count = r.u64()?;
    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = r.string()?;
        let ty = r.u32()?;
        if let Some(v) = r.skip_value(ty)? {
            metadata.insert(key, v);
        }
    }
    Ok(GgufInfo {
        version,
        tensor_count,
        architecture: metadata.get("general.architecture").cloned(),
        name: metadata.get("general.name").cloned(),
        metadata,
    })
}

/// Parse a GGUF header from a file without loading the whole
/// (potentially multi-GB) model: reads the first 16 MiB, and only
/// falls back to a full read if the header itself is larger.
pub fn parse_gguf_file(path: &Path) -> Result<GgufInfo, ModelError> {
    use std::io::Read as _;
    // Grow the header window incrementally up to a 64 MiB cap; a
    // truncated or oversized header errors without ever reading a
    // multi-GB tensor payload into RAM.
    const HEADER_CAP: u64 = 64 * 1024 * 1024;
    let mut file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    let mut window: u64 = 16 * 1024 * 1024;
    loop {
        buf.clear();
        file.seek(std::io::SeekFrom::Start(0))?;
        (&mut file).take(window).read_to_end(&mut buf)?;
        match parse_gguf(&buf) {
            Err(ModelError::Truncated(_))
                if (buf.len() as u64) >= window && window < HEADER_CAP =>
            {
                window = (window * 2).min(HEADER_CAP);
            }
            other => return other,
        }
    }
}

use std::io::Seek as _;

/// Unique temp file in the cache dir (O_EXCL via create_new, so a
/// pre-planted symlink at a predictable name is never followed).
fn unique_tmp(dir: &Path, name: &str) -> Result<PathBuf, ModelError> {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    for _ in 0..16 {
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let candidate = dir.join(format!(".{name}.{}-{n}.tmp", std::process::id()));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(_) => return Ok(candidate),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Err(ModelError::Http(
        "could not create a unique temp file".into(),
    ))
}

pub fn model_cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("GRIDTALK_MODEL_CACHE") {
        if !d.is_empty() {
            return PathBuf::from(d);
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache/gridtalk/models")
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelEntry {
    pub name: String,
    pub path: PathBuf,
    pub bytes: u64,
    pub info: Option<GgufInfo>,
}

pub struct ModelManager {
    pub dir: PathBuf,
}

impl ModelManager {
    pub fn new(dir: PathBuf) -> Self {
        ModelManager { dir }
    }

    pub fn default_cache() -> Self {
        ModelManager::new(model_cache_dir())
    }

    pub fn list(&self) -> Vec<ModelEntry> {
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension()
                    .and_then(|x| x.to_str())
                    .is_some_and(|x| x.eq_ignore_ascii_case("gguf"))
                {
                    let bytes = e.metadata().map(|m| m.len()).unwrap_or(0);
                    let info = parse_gguf_file(&p).ok();
                    out.push(ModelEntry {
                        name: p
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned(),
                        path: p,
                        bytes,
                        info,
                    });
                }
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    fn checked_name(name: &str) -> Result<String, ModelError> {
        // Reject path traversal (separators, leading dot-dot) but
        // allow interior dots, e.g. "model..v2.gguf".
        if name.is_empty()
            || name.contains('/')
            || name.contains('\\')
            || name == ".."
            || name.starts_with("../")
        {
            return Err(ModelError::InvalidName(name.into()));
        }
        let lower = name.to_ascii_lowercase();
        let n = if lower.ends_with(".gguf") {
            name.to_string()
        } else {
            format!("{name}.gguf")
        };
        Ok(n)
    }

    /// Copy a local GGUF file into the cache after validating its
    /// header. Copies to a temp file and renames, so a crash never
    /// leaves a half-written model in the cache.
    pub fn import(&self, src: &Path) -> Result<ModelEntry, ModelError> {
        let info = parse_gguf_file(src)?;
        std::fs::create_dir_all(&self.dir)?;
        let file_name = src
            .file_name()
            .ok_or_else(|| ModelError::InvalidName(src.display().to_string()))?
            .to_string_lossy()
            .into_owned();
        let name = Self::checked_name(&file_name)?;
        let dest = self.dir.join(&name);
        let tmp = unique_tmp(&self.dir, &name)?;
        let copy_result = (|| -> Result<u64, ModelError> {
            let mut from = std::fs::File::open(src)?;
            let mut to = std::fs::File::create(&tmp)?;
            let n = std::io::copy(&mut from, &mut to)?;
            // Re-validate the copied bytes, not just the source
            // (the source may have changed during the copy).
            parse_gguf_file(&tmp)?;
            Ok(n)
        })();
        let bytes = match copy_result {
            Ok(n) => n,
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                return Err(e);
            }
        };
        std::fs::rename(&tmp, &dest)?;
        Ok(ModelEntry {
            name,
            path: dest,
            bytes,
            info: Some(info),
        })
    }

    pub fn remove(&self, name: &str) -> Result<(), ModelError> {
        let name = Self::checked_name(name)?;
        let path = self.dir.join(name);
        std::fs::remove_file(path)?;
        Ok(())
    }

    /// Download a GGUF over plain http (local mirrors / proxies),
    /// streamed to a temp file (never held whole in RAM) and
    /// renamed into place only after its header validates.
    /// https is refused until a TLS transport lands.
    pub fn download_http(&self, url: &str, name: &str) -> Result<ModelEntry, ModelError> {
        if url.contains(['\r', '\n', ' ']) {
            return Err(ModelError::Http(
                "url contains whitespace/control characters".into(),
            ));
        }
        if url.starts_with("https://") {
            return Err(ModelError::Http(
                "https model URLs need a TLS transport or local proxy; none configured".into(),
            ));
        }
        let rest = url
            .strip_prefix("http://")
            .ok_or_else(|| ModelError::Http("url must start with http://".into()))?;
        let (host_port, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };
        use std::net::ToSocketAddrs as _;
        let addr = host_port
            .to_socket_addrs()
            .map_err(|e| ModelError::Http(format!("cannot resolve {host_port}: {e}")))?
            .next()
            .ok_or_else(|| ModelError::Http(format!("cannot resolve {host_port}")))?;
        let mut stream =
            std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(10))?;
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(60)))
            .ok();
        let req = format!(
            "GET {path} HTTP/1.1\r\nHost: {host_port}\r\nConnection: close\r\nUser-Agent: GridTalk\r\n\r\n"
        );
        stream.write_all(req.as_bytes())?;
        // Read headers byte-wise until CRLFCRLF.
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            let n = stream.read(&mut byte)?;
            if n == 0 {
                return Err(ModelError::Http("connection closed in headers".into()));
            }
            head.push(byte[0]);
            if head.len() > 64 * 1024 {
                return Err(ModelError::Http("headers too large".into()));
            }
        }
        let head_text = String::from_utf8_lossy(&head).into_owned();
        let status: u32 = head_text
            .split_whitespace()
            .nth(1)
            .and_then(|code| code.parse().ok())
            .unwrap_or(0);
        if status != 200 {
            return Err(ModelError::Http(format!(
                "unexpected status: {}",
                head_text.lines().next().unwrap_or("")
            )));
        }
        if head_text
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked")
        {
            return Err(ModelError::Http(
                "chunked download responses are not supported; use a mirror that sends Content-Length".into(),
            ));
        }
        std::fs::create_dir_all(&self.dir)?;
        let name = Self::checked_name(name)?;
        let tmp = unique_tmp(&self.dir, &name)?;
        // Stream with a hard size cap (64 GiB) and a total deadline
        // (2 h); the temp file is deleted on any breach.
        const MAX_DOWNLOAD_BYTES: u64 = 64 * 1024 * 1024 * 1024;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2 * 60 * 60);
        let copied = (|| -> Result<u64, ModelError> {
            // unique_tmp already created the file (O_EXCL).
            let mut out = std::fs::OpenOptions::new().write(true).open(&tmp)?;
            let mut limited = (&mut stream).take(MAX_DOWNLOAD_BYTES + 1);
            let mut buf = [0u8; 64 * 1024];
            let mut total: u64 = 0;
            loop {
                let n = limited.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                if std::time::Instant::now() > deadline {
                    return Err(ModelError::Http(
                        "download exceeded the 2-hour deadline".into(),
                    ));
                }
                out.write_all(&buf[..n])?;
                total += n as u64;
                if total > MAX_DOWNLOAD_BYTES {
                    return Err(ModelError::Http("download exceeds the 64 GiB cap".into()));
                }
            }
            Ok(total)
        })();
        let copied = match copied {
            Ok(n) => n,
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                return Err(e);
            }
        };
        let info = parse_gguf_file(&tmp).inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp);
        })?;
        let dest = self.dir.join(&name);
        std::fs::rename(&tmp, &dest)?;
        Ok(ModelEntry {
            name,
            path: dest,
            bytes: copied,
            info: Some(info),
        })
    }
}

#[cfg(test)]
pub fn synthetic_gguf(arch: &str) -> Vec<u8> {
    // Minimal valid GGUF v3: 2 metadata KVs, 0 tensors.
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes()); // tensors
    b.extend_from_slice(&2u64.to_le_bytes()); // kv count
    let kv = |b: &mut Vec<u8>, k: &str, v: &str| {
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k.as_bytes());
        b.extend_from_slice(&8u32.to_le_bytes()); // string
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        b.extend_from_slice(v.as_bytes());
    };
    kv(&mut b, "general.architecture", arch);
    kv(&mut b, "general.name", "Synthetic Test Model");
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_synthetic_gguf() {
        let info = parse_gguf(&synthetic_gguf("qwen3moe")).unwrap();
        assert_eq!(info.version, 3);
        assert_eq!(info.architecture.as_deref(), Some("qwen3moe"));
        assert_eq!(info.name.as_deref(), Some("Synthetic Test Model"));
        assert!(parse_gguf(b"NOPE").is_err());
    }

    #[test]
    fn regression_gguf_nesting_overflow_and_names() {
        // Deeply nested array metadata must error, not overflow the stack.
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        let key = "general.architecture";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&9u32.to_le_bytes()); // array
        for _ in 0..40 {
            b.extend_from_slice(&9u32.to_le_bytes()); // elem type: array
            b.extend_from_slice(&1u64.to_le_bytes()); // count 1
        }
        b.extend_from_slice(&8u32.to_le_bytes());
        assert!(parse_gguf(&b).is_err());
        // Interior dots in names are fine; traversal is not.
        assert!(ModelManager::checked_name("model..v2").is_ok());
        assert!(ModelManager::checked_name("MODEL.GGUF").is_ok());
        assert!(ModelManager::checked_name("../evil").is_err());
        assert!(ModelManager::checked_name("a/b").is_err());
    }

    #[test]
    fn manager_import_list_remove() {
        let dir = std::env::temp_dir().join(format!("gt-models-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mgr = ModelManager::new(dir.join("cache"));
        let src_dir = dir.join("downloads");
        std::fs::create_dir_all(&src_dir).unwrap();
        let src = src_dir.join("src.gguf");
        std::fs::write(&src, synthetic_gguf("llama")).unwrap();
        let entry = mgr.import(&src).unwrap();
        assert_eq!(entry.info.unwrap().architecture.as_deref(), Some("llama"));
        assert_eq!(mgr.list().len(), 1);
        mgr.remove("src").unwrap();
        assert!(mgr.list().is_empty());
        assert!(mgr.remove("../evil").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
