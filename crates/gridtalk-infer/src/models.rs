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
    /// Offset where tensor payload begins (after infos, aligned).
    pub data_start: u64,
    pub architecture: Option<String>,
    pub name: Option<String>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug)]
pub enum ModelError {
    NotGguf(String),
    Truncated(String),
    /// Tensor extent beyond EOF: not a header truncation, so the
    /// windowed header reader must not retry on it.
    Extent(String),
    Io(std::io::Error),
    Http(String),
    InvalidName(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::NotGguf(m) => write!(f, "not a GGUF file: {m}"),
            ModelError::Truncated(m) => write!(f, "truncated GGUF: {m}"),
            ModelError::Extent(m) => write!(f, "GGUF tensor extent invalid: {m}"),
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
            0 | 1 => {
                self.take(1)?;
                Ok(None)
            }
            7 => {
                let b = self.take(1)?[0] != 0;
                Ok(Some(b.to_string()))
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
                let mut parts = Vec::new();
                for _ in 0..count {
                    if let Some(v) = self.skip_value_depth(elem, depth + 1)? {
                        if parts.len() < 8 {
                            parts.push(v);
                        }
                    }
                }
                if parts.is_empty() {
                    Ok(Some(format!("[{count} items]")))
                } else {
                    Ok(Some(format!("[{}]", parts.join(", "))))
                }
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

/// (block elements, bytes per block) for known GGML tensor types.
fn ggml_type_layout(ty: u32) -> Option<(u64, u64)> {
    // Enum order and block sizes from the vendored ggml.h /
    // ggml-common.h (llama-cpp-sys-2 0.1.160), cross-checked
    // against its static_assert block sizes (QK_K = 256).
    Some(match ty {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 36),    // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 50),  // IQ1_S
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        29 => (256, 56),  // IQ1_M
        30 => (1, 2),     // BF16
        34 => (256, 54),  // TQ1_0
        35 => (256, 66),  // TQ2_0
        39 => (32, 17),   // MXFP4
        40 => (64, 36),   // NVFP4
        41 => (128, 18),  // Q1_0
        42 => (64, 18),   // Q2_0
        _ => return None,
    })
}

/// Parse a GGUF header (v2/v3 layout) without loading tensors.
pub fn parse_gguf(bytes: &[u8]) -> Result<GgufInfo, ModelError> {
    parse_gguf_with_len(bytes, bytes.len() as u64)
}

fn parse_gguf_with_len(bytes: &[u8], file_len: u64) -> Result<GgufInfo, ModelError> {
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
    // Tensor infos must actually be present: name, dims, type,
    // offset for every declared tensor. A header claiming tensors
    // with no infos is truncated, not a model.
    let mut tensor_extents: Vec<(String, u64, u64)> = Vec::new(); // (name, offset, bytes)
    for _ in 0..tensor_count {
        let name = r.string()?;
        let n_dims = r.u32()?;
        if n_dims > 8 {
            return Err(ModelError::NotGguf(format!(
                "tensor has {n_dims} dimensions"
            )));
        }
        let mut elements: u64 = 1;
        for _ in 0..n_dims {
            let d = r.u64()?;
            elements = elements
                .checked_mul(d)
                .ok_or_else(|| ModelError::NotGguf("tensor dimension overflow".into()))?;
        }
        let ty = r.u32()?;
        let offset = r.u64()?;
        // Unknown (newer) GGML types: extent cannot be verified
        // here; llama.cpp remains the authority for those.
        let size = match ggml_type_layout(ty) {
            Some((block, per_block)) => Some(
                elements
                    .div_ceil(block)
                    .checked_mul(per_block)
                    .ok_or_else(|| ModelError::NotGguf(format!("tensor {name} size overflow")))?,
            ),
            None => None,
        };
        tensor_extents.push((name, offset, size.unwrap_or(0)));
    }
    // gguf.cpp rejects zero / non-power-of-two alignment outright;
    // mirror that instead of silently substituting 32. Any
    // power-of-two is accepted (checked_mul guards overflow).
    let alignment: u64 = match metadata.get("general.alignment") {
        Some(v) => v
            .parse::<u64>()
            .ok()
            .filter(|a| a.is_power_of_two())
            .ok_or_else(|| ModelError::NotGguf(format!("invalid general.alignment {v:?}")))?,
        None => 32,
    };
    let data_start = (r.pos as u64)
        .div_ceil(alignment)
        .checked_mul(alignment)
        .ok_or_else(|| ModelError::NotGguf("GGUF data offset overflow".into()))?;
    // A data_start within one alignment quantum of EOF is the
    // usual header padding; far beyond EOF (e.g. a 2^40 alignment)
    // is bogus even with zero tensors.
    if data_start > file_len && data_start - file_len >= alignment.max(1) {
        return Err(ModelError::Truncated(
            "GGUF data start lies beyond the end of the file".into(),
        ));
    }
    if tensor_count > 0 && file_len <= data_start {
        return Err(ModelError::Truncated(
            "GGUF declares tensors but has no tensor payload".into(),
        ));
    }
    // Every verifiable tensor must lie inside the actual file.
    for (name, offset, size) in &tensor_extents {
        let end = data_start
            .checked_add(*offset)
            .and_then(|v| v.checked_add(*size))
            .ok_or_else(|| ModelError::NotGguf(format!("tensor {name} extent overflow")))?;
        if end > file_len {
            return Err(ModelError::Extent(format!(
                "tensor {name} extends to {end}, beyond the {file_len}-byte file"
            )));
        }
    }
    Ok(GgufInfo {
        version,
        tensor_count,
        data_start,
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
    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
    let mut buf = Vec::new();
    let mut window: u64 = 1024 * 1024;
    loop {
        buf.clear();
        file.seek(std::io::SeekFrom::Start(0))?;
        (&mut file).take(window).read_to_end(&mut buf)?;
        match parse_gguf_with_len(&buf, file_len) {
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

/// Best-effort directory fsync (durability after rename).
fn sync_dir(dir: &Path) {
    #[cfg(unix)]
    if let Ok(d) = std::fs::File::open(dir) {
        let _ = d.sync_all();
    }
    #[cfg(not(unix))]
    let _ = dir;
}

/// Unique temp file in the cache dir (O_EXCL via create_new, so a
/// pre-planted symlink at a predictable name is never followed).
fn unique_tmp(dir: &Path, name: &str) -> Result<(PathBuf, std::fs::File), ModelError> {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    for _ in 0..16 {
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let candidate = dir.join(format!(".{name}.{}-{n}.tmp", std::process::id()));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            // Return the live handle: callers write through it and
            // never reopen the temp by path (which would void the
            // O_EXCL protection).
            Ok(file) => return Ok((candidate, file)),
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
    // Platform cache dir: %LOCALAPPDATA% on Windows, $XDG_CACHE_HOME
    // or ~/.cache elsewhere.
    if let Ok(d) = std::env::var("LOCALAPPDATA") {
        if !d.is_empty() {
            return PathBuf::from(d).join("gridtalk/models");
        }
    }
    if let Ok(d) = std::env::var("XDG_CACHE_HOME") {
        if !d.is_empty() {
            return PathBuf::from(d).join("gridtalk/models");
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
        parse_gguf_file(src)?; // validate the source up front
        std::fs::create_dir_all(&self.dir)?;
        let file_name = src
            .file_name()
            .ok_or_else(|| ModelError::InvalidName(src.display().to_string()))?
            .to_string_lossy()
            .into_owned();
        let name = Self::checked_name(&file_name)?;
        let dest = self.dir.join(&name);
        let (tmp, mut tmp_file) = unique_tmp(&self.dir, &name)?;
        let copy_result = (|| -> Result<(u64, GgufInfo), ModelError> {
            let mut from = std::fs::File::open(src)?;
            let n = std::io::copy(&mut from, &mut tmp_file)?;
            tmp_file.sync_all()?;
            drop(tmp_file);
            // Re-validate the copied bytes, not just the source
            // (the source may have changed during the copy), and
            // return the COPY's info, not the pre-copy parse.
            let copied_info = parse_gguf_file(&tmp)?;
            Ok((n, copied_info))
        })();
        let (bytes, info) = match copy_result {
            Ok(pair) => pair,
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                return Err(e);
            }
        };
        std::fs::rename(&tmp, &dest).inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp);
        })?;
        sync_dir(&self.dir);
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
        // The total deadline covers connect and headers too.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2 * 60 * 60);
        use std::net::ToSocketAddrs as _;
        let addrs: Vec<std::net::SocketAddr> = host_port
            .to_socket_addrs()
            .map_err(|e| ModelError::Http(format!("cannot resolve {host_port}: {e}")))?
            .collect();
        let mut stream = None;
        let mut last_err: Option<std::io::Error> = None;
        for addr in &addrs {
            match std::net::TcpStream::connect_timeout(addr, std::time::Duration::from_secs(10)) {
                Ok(s) => {
                    stream = Some(s);
                    break;
                }
                Err(e) => last_err = Some(e),
            }
        }
        let mut stream = stream.ok_or_else(|| {
            ModelError::Http(format!(
                "cannot connect to {host_port}: {}",
                last_err.map(|e| e.to_string()).unwrap_or_default()
            ))
        })?;
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
            if std::time::Instant::now() > deadline {
                return Err(ModelError::Http(
                    "download exceeded the 2-hour deadline during headers".into(),
                ));
            }
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
        // Expected size, when the server sends one: a short read
        // is a truncated model, never a successful download.
        let content_length: Option<u64> = head_text.lines().find_map(|l| {
            let (name, value) = l.split_once(':')?;
            if name.trim().eq_ignore_ascii_case("content-length") {
                value.trim().parse().ok()
            } else {
                None
            }
        });
        std::fs::create_dir_all(&self.dir)?;
        let name = Self::checked_name(name)?;
        let (tmp, mut out) = unique_tmp(&self.dir, &name)?;
        // Stream with a hard size cap (64 GiB) and the total
        // deadline above; the temp file is deleted on any breach.
        const MAX_DOWNLOAD_BYTES: u64 = 64 * 1024 * 1024 * 1024;
        let copied = (|| -> Result<u64, ModelError> {
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
                if total % (256 * 1024 * 1024) < n as u64 {
                    eprintln!("gridtalk: downloaded {total} bytes…");
                }
                if total > MAX_DOWNLOAD_BYTES {
                    return Err(ModelError::Http("download exceeds the 64 GiB cap".into()));
                }
            }
            out.sync_all()?;
            Ok(total)
        })();
        let copied = match copied {
            Ok(n) => {
                if let Some(expected) = content_length {
                    if n != expected {
                        let _ = std::fs::remove_file(&tmp);
                        return Err(ModelError::Http(format!(
                            "truncated download: got {n} of {expected} bytes"
                        )));
                    }
                }
                n
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                return Err(e);
            }
        };
        let info = parse_gguf_file(&tmp).inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp);
        })?;
        let dest = self.dir.join(&name);
        std::fs::rename(&tmp, &dest).inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp);
        })?;
        sync_dir(&self.dir);
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
    fn gguf_tensor_extent_beyond_eof_is_rejected() {
        // One F32 tensor of 10 elements declared, but no payload.
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        b.extend_from_slice(&0u64.to_le_bytes()); // 0 kv
        let name = "w";
        b.extend_from_slice(&(name.len() as u64).to_le_bytes());
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&1u32.to_le_bytes()); // 1 dim
        b.extend_from_slice(&10u64.to_le_bytes()); // 10 elements
        b.extend_from_slice(&0u32.to_le_bytes()); // F32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset 0
        assert!(parse_gguf(&b).is_err());
        // With the 40 payload bytes present it parses.
        let mut with_payload = b.clone();
        // Pad to the 32-byte-aligned data_start, then payload.
        let aligned = (with_payload.len() as u64).div_ceil(32) * 32;
        with_payload.resize(aligned as usize, 0);
        with_payload.extend_from_slice(&[0u8; 40]);
        assert!(parse_gguf(&with_payload).is_ok());
    }

    #[test]
    fn gguf_iq_type_sizes_use_the_ggml_enum() {
        // Type 16 is IQ2_XXS (256 elements / 66-byte blocks), NOT
        // I8: a 256-element tensor needs exactly 66 payload bytes.
        let build = |ty: u32, payload: usize| -> Vec<u8> {
            let mut b = Vec::new();
            b.extend_from_slice(b"GGUF");
            b.extend_from_slice(&3u32.to_le_bytes());
            b.extend_from_slice(&1u64.to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes());
            b.extend_from_slice(&1u64.to_le_bytes());
            b.extend_from_slice(b"w");
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&256u64.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes());
            let aligned = (b.len() as u64).div_ceil(32) * 32;
            b.resize(aligned as usize, 0);
            b.extend_from_slice(&vec![0u8; payload]);
            b
        };
        assert!(parse_gguf(&build(16, 10)).is_err());
        assert!(parse_gguf(&build(16, 66)).is_ok());
        // Type 24 is I8: 256 elements need 256 bytes.
        assert!(parse_gguf(&build(24, 66)).is_err());
        assert!(parse_gguf(&build(24, 256)).is_ok());
    }

    #[test]
    fn gguf_q1_q2_nvfp4_layouts() {
        // Q1_0: 128 elems/18-byte blocks; Q2_0: 64/18; NVFP4: 64/36.
        let build = |ty: u32, elems: u64, payload: usize| -> Vec<u8> {
            let mut b = Vec::new();
            b.extend_from_slice(b"GGUF");
            b.extend_from_slice(&3u32.to_le_bytes());
            b.extend_from_slice(&1u64.to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes());
            b.extend_from_slice(&1u64.to_le_bytes());
            b.extend_from_slice(b"w");
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&elems.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes());
            let aligned = (b.len() as u64).div_ceil(32) * 32;
            b.resize(aligned as usize, 0);
            b.extend_from_slice(&vec![0u8; payload]);
            b
        };
        assert!(parse_gguf(&build(41, 128, 17)).is_err());
        assert!(parse_gguf(&build(41, 128, 18)).is_ok());
        assert!(parse_gguf(&build(42, 64, 17)).is_err());
        assert!(parse_gguf(&build(42, 64, 18)).is_ok());
        assert!(parse_gguf(&build(40, 64, 35)).is_err());
        assert!(parse_gguf(&build(40, 64, 36)).is_ok());
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
