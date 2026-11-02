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
        if self.pos + n > self.b.len() {
            return Err(ModelError::Truncated(format!(
                "need {n} bytes at {} of {}",
                self.pos,
                self.b.len()
            )));
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
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
                for _ in 0..count {
                    self.skip_value(elem)?;
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
                if p.extension().and_then(|x| x.to_str()) == Some("gguf") {
                    let bytes = e.metadata().map(|m| m.len()).unwrap_or(0);
                    let info = std::fs::read(&p).ok().and_then(|b| parse_gguf(&b).ok());
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
        if name.is_empty() || name.contains('/') || name.contains('\\') || name.contains("..") {
            return Err(ModelError::InvalidName(name.into()));
        }
        let n = if name.ends_with(".gguf") {
            name.to_string()
        } else {
            format!("{name}.gguf")
        };
        Ok(n)
    }

    /// Copy a local GGUF file into the cache after validating it.
    pub fn import(&self, src: &Path) -> Result<ModelEntry, ModelError> {
        let bytes = std::fs::read(src)?;
        let info = parse_gguf(&bytes)?;
        std::fs::create_dir_all(&self.dir)?;
        let file_name = src
            .file_name()
            .ok_or_else(|| ModelError::InvalidName(src.display().to_string()))?
            .to_string_lossy()
            .into_owned();
        let name = Self::checked_name(&file_name)?;
        let dest = self.dir.join(&name);
        std::fs::write(&dest, &bytes)?;
        Ok(ModelEntry {
            name,
            path: dest,
            bytes: bytes.len() as u64,
            info: Some(info),
        })
    }

    pub fn remove(&self, name: &str) -> Result<(), ModelError> {
        let name = Self::checked_name(name)?;
        let path = self.dir.join(name);
        std::fs::remove_file(path)?;
        Ok(())
    }

    /// Download a GGUF over plain http (local mirrors / proxies).
    /// https is refused with a typed error until a TLS transport
    /// lands — never silently downgraded.
    pub fn download_http(&self, url: &str, name: &str) -> Result<ModelEntry, ModelError> {
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
        let mut stream = std::net::TcpStream::connect(host_port)?;
        let req = format!(
            "GET {path} HTTP/1.1\r\nHost: {host_port}\r\nConnection: close\r\nUser-Agent: GridTalk\r\n\r\n"
        );
        stream.write_all(req.as_bytes())?;
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw)?;
        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or_else(|| ModelError::Http("malformed HTTP response".into()))?;
        let head = String::from_utf8_lossy(&raw[..split]).into_owned();
        if !head.contains(" 200") {
            return Err(ModelError::Http(format!(
                "unexpected status: {}",
                head.lines().next().unwrap_or("")
            )));
        }
        let body = &raw[split + 4..];
        let info = parse_gguf(body)?;
        std::fs::create_dir_all(&self.dir)?;
        let name = Self::checked_name(name)?;
        let dest = self.dir.join(&name);
        std::fs::write(&dest, body)?;
        Ok(ModelEntry {
            name,
            path: dest,
            bytes: body.len() as u64,
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
