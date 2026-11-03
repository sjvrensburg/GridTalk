//! Shared client transport for the OpenAI-compatible backend and
//! model downloads: URL parsing, TCP connect with timeouts, and an
//! optional TLS layer (rustls with the `ring` provider, verified
//! against the bundled Mozilla roots plus an optional extra CA file).
//!
//! Extra trust anchors (a corporate CA, a self-signed local server)
//! come from a PEM file named by `GRIDTALK_CA_FILE`. Certificate
//! verification is never disabled.

use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, ServerName};
use rustls::{ClientConfig, ClientConnection, RootCertStore, StreamOwned};
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// Environment variable naming a PEM file of extra trusted CAs.
pub const CA_FILE_ENV: &str = "GRIDTALK_CA_FILE";

/// A parsed `http://` or `https://` URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Url {
    pub tls: bool,
    /// Host without IPv6 brackets (for DNS and TLS SNI).
    pub host: String,
    pub port: u16,
    /// `host[:port]` exactly as written (for the Host header).
    pub authority: String,
    /// Path and query; empty when the URL has none.
    pub path: String,
}

impl Url {
    pub fn scheme(&self) -> &'static str {
        if self.tls {
            "https"
        } else {
            "http"
        }
    }
}

/// Parse an absolute `http://` or `https://` URL. Default ports
/// are 80 and 443. Userinfo (`user:pass@`) is rejected so
/// credentials never travel in a URL.
pub fn parse_url(url: &str) -> Result<Url, String> {
    if url.contains(|c: char| c.is_whitespace() || c.is_control()) {
        return Err("URL contains whitespace or control characters".into());
    }
    let (tls, rest) = if let Some(rest) = url.strip_prefix("https://") {
        (true, rest)
    } else if let Some(rest) = url.strip_prefix("http://") {
        (false, rest)
    } else {
        return Err(format!("URL must start with http:// or https://: {url}"));
    };
    let (authority, path) = match rest.find(['/', '?', '#']) {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, ""),
    };
    // Fragments are client-side only.
    let path = path.split('#').next().unwrap_or("");
    let path = if path.starts_with('?') {
        format!("/{path}")
    } else {
        path.to_string()
    };
    if authority.contains('@') {
        return Err("credentials in URLs are not supported".into());
    }
    let default_port = if tls { 443 } else { 80 };
    let (host, port) = if let Some(after) = authority.strip_prefix('[') {
        let (host, tail) = after
            .split_once(']')
            .ok_or_else(|| format!("unterminated IPv6 address in {url}"))?;
        let port = match tail {
            "" => default_port,
            t => t
                .strip_prefix(':')
                .and_then(|p| p.parse().ok())
                .ok_or_else(|| format!("bad port in {url}"))?,
        };
        (host.to_string(), port)
    } else {
        match authority.rsplit_once(':') {
            Some((h, p)) => (
                h.to_string(),
                p.parse().map_err(|_| format!("bad port in {url}"))?,
            ),
            None => (authority.to_string(), default_port),
        }
    };
    if host.is_empty() {
        return Err(format!("URL has no host: {url}"));
    }
    Ok(Url {
        tls,
        host,
        port,
        authority: authority.to_string(),
        path,
    })
}

/// Resolve a redirect `Location` against the URL that returned it
/// (RFC 3986 reference resolution): absolute, scheme-relative
/// (`//host/...`), absolute-path (`/...`), query-only (`?...`) and
/// relative-path (`file`, `../dir/file`) forms.
pub fn resolve_location(base: &Url, location: &str) -> Result<Url, String> {
    let location = location.trim();
    let lower = location.to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") {
        return parse_url(location);
    }
    if let Some(rest) = location.strip_prefix("//") {
        return parse_url(&format!("{}://{rest}", base.scheme()));
    }
    if location.is_empty() || location.starts_with('#') {
        return Err(format!("unsupported redirect location {location:?}"));
    }
    // Fragments never go on the wire.
    let location = location.split('#').next().unwrap_or_default();
    let base_path = match base.path.split('?').next() {
        Some(p) if !p.is_empty() => p,
        _ => "/",
    };
    let (path, query) = match location.split_once('?') {
        Some((p, q)) => (p, Some(q)),
        None => (location, None),
    };
    let merged = if path.is_empty() {
        // `?query` keeps the base path.
        base_path.to_string()
    } else if path.starts_with('/') {
        path.to_string()
    } else {
        let dir = &base_path[..=base_path.rfind('/').unwrap_or(0)];
        format!("{dir}{path}")
    };
    let mut target = remove_dot_segments(&merged);
    if let Some(q) = query {
        target.push('?');
        target.push_str(q);
    }
    parse_url(&format!("{}://{}{target}", base.scheme(), base.authority))
}

/// Resolve `.` and `..` segments of an absolute path; `..` never
/// climbs above the root.
fn remove_dot_segments(path: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    let segments: Vec<&str> = path.split('/').skip(1).collect();
    let last = segments.len().saturating_sub(1);
    for (i, seg) in segments.iter().enumerate() {
        match *seg {
            "." | ".." => {
                if *seg == ".." {
                    out.pop();
                }
                // A trailing dot segment names the directory itself.
                if i == last {
                    out.push("");
                }
            }
            s => out.push(s),
        }
    }
    format!("/{}", out.join("/"))
}

/// Decodes an HTTP/1.1 chunked body while streaming it: yields the
/// chunk data, consumes the size lines, CRLFs and trailers, and
/// reports EOF only after the terminating zero-size chunk. A
/// connection that closes before that is an `UnexpectedEof` error,
/// so a truncated body is never mistaken for a complete one.
pub struct ChunkedReader<R> {
    inner: R,
    /// Bytes left in the current chunk.
    remaining: u64,
    done: bool,
}

/// Longest size or trailer line accepted.
const CHUNK_LINE_MAX: usize = 4096;

impl<R: Read> ChunkedReader<R> {
    pub fn new(inner: R) -> Self {
        ChunkedReader {
            inner,
            remaining: 0,
            done: false,
        }
    }

    fn read_line(&mut self) -> io::Result<String> {
        let mut line = Vec::new();
        let mut byte = [0u8; 1];
        while !line.ends_with(b"\r\n") {
            if self.inner.read(&mut byte)? == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "connection closed inside a chunked body",
                ));
            }
            line.push(byte[0]);
            if line.len() > CHUNK_LINE_MAX {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "chunked encoding line too long",
                ));
            }
        }
        line.truncate(line.len() - 2);
        Ok(String::from_utf8_lossy(&line).into_owned())
    }
}

impl<R: Read> Read for ChunkedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.done || buf.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            let line = self.read_line()?;
            let size = line.split(';').next().unwrap_or_default().trim();
            self.remaining = u64::from_str_radix(size, 16).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("bad chunk size line {line:?}"),
                )
            })?;
            if self.remaining == 0 {
                // Skip trailers up to the final blank line.
                while !self.read_line()?.is_empty() {}
                self.done = true;
                return Ok(0);
            }
        }
        let want = buf
            .len()
            .min(usize::try_from(self.remaining).unwrap_or(usize::MAX));
        let n = self.inner.read(&mut buf[..want])?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "connection closed inside a chunk",
            ));
        }
        self.remaining -= n as u64;
        if self.remaining == 0 && !self.read_line()?.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "missing CRLF after chunk data",
            ));
        }
        Ok(n)
    }
}

/// A plain or TLS connection.
pub enum Conn {
    Plain(TcpStream),
    Tls(Box<StreamOwned<ClientConnection, TcpStream>>),
}

impl Conn {
    fn tcp(&self) -> &TcpStream {
        match self {
            Conn::Plain(s) => s,
            Conn::Tls(s) => &s.sock,
        }
    }

    pub fn set_read_timeout(&self, d: Option<Duration>) -> io::Result<()> {
        self.tcp().set_read_timeout(d)
    }
}

impl Read for Conn {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Conn::Plain(s) => s.read(buf),
            // Many servers close without a TLS close_notify. Treat
            // that as end of stream: callers check message framing
            // (Content-Length, chunked terminator) themselves.
            Conn::Tls(s) => match s.read(buf) {
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(0),
                other => other,
            },
        }
    }
}

impl Write for Conn {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Conn::Plain(s) => s.write(buf),
            Conn::Tls(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self {
            Conn::Plain(s) => s.flush(),
            Conn::Tls(s) => s.flush(),
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Test-only trust override (a self-signed local server).
    pub static TEST_TLS_CONFIG: std::cell::RefCell<Option<Arc<ClientConfig>>> =
        const { std::cell::RefCell::new(None) };
}

fn build_tls_config() -> Result<Arc<ClientConfig>, String> {
    let mut roots = RootCertStore {
        roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
    };
    if let Ok(path) = std::env::var(CA_FILE_ENV) {
        if !path.is_empty() {
            let mut added = 0usize;
            let certs = CertificateDer::pem_file_iter(&path)
                .map_err(|e| format!("{CA_FILE_ENV}={path}: {e}"))?;
            for cert in certs {
                let cert = cert.map_err(|e| format!("{CA_FILE_ENV}={path}: {e}"))?;
                roots
                    .add(cert)
                    .map_err(|e| format!("{CA_FILE_ENV}={path}: {e}"))?;
                added += 1;
            }
            if added == 0 {
                return Err(format!("{CA_FILE_ENV}={path}: no certificates found"));
            }
        }
    }
    let config =
        ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()
            .map_err(|e| e.to_string())?
            .with_root_certificates(roots)
            .with_no_client_auth();
    Ok(Arc::new(config))
}

fn tls_config() -> io::Result<Arc<ClientConfig>> {
    #[cfg(test)]
    if let Some(cfg) = TEST_TLS_CONFIG.with(|c| c.borrow().clone()) {
        return Ok(cfg);
    }
    static CONFIG: OnceLock<Result<Arc<ClientConfig>, String>> = OnceLock::new();
    CONFIG
        .get_or_init(build_tls_config)
        .clone()
        .map_err(io::Error::other)
}

/// Connect to `url`'s host, trying each resolved address with
/// `connect_timeout`. For `https`, the TLS handshake completes
/// here (bounded by `io_timeout`), so certificate errors surface
/// as connect errors rather than on the first read.
pub fn connect(url: &Url, connect_timeout: Duration, io_timeout: Duration) -> io::Result<Conn> {
    let addrs: Vec<SocketAddr> = (url.host.as_str(), url.port)
        .to_socket_addrs()
        .map_err(|e| io::Error::new(e.kind(), format!("cannot resolve {}: {e}", url.host)))?
        .collect();
    let mut last: Option<io::Error> = None;
    let mut tcp = None;
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, connect_timeout) {
            Ok(s) => {
                tcp = Some(s);
                break;
            }
            Err(e) => last = Some(e),
        }
    }
    let tcp = tcp.ok_or_else(|| {
        let detail = last
            .map(|e| e.to_string())
            .unwrap_or_else(|| "no addresses".into());
        io::Error::new(
            io::ErrorKind::ConnectionRefused,
            format!("cannot connect to {}: {detail}", url.authority),
        )
    })?;
    tcp.set_read_timeout(Some(io_timeout))?;
    tcp.set_write_timeout(Some(io_timeout))?;
    if !url.tls {
        return Ok(Conn::Plain(tcp));
    }
    let name = ServerName::try_from(url.host.clone()).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid TLS server name {}: {e}", url.host),
        )
    })?;
    let conn = ClientConnection::new(tls_config()?, name).map_err(io::Error::other)?;
    let mut stream = StreamOwned::new(conn, tcp);
    while stream.conn.is_handshaking() {
        stream.conn.complete_io(&mut stream.sock).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("TLS handshake with {} failed: {e}", url.authority),
            )
        })?;
    }
    Ok(Conn::Tls(Box::new(stream)))
}

#[cfg(test)]
pub mod test_support {
    //! A one-shot local HTTPS server with a self-signed certificate.
    use super::*;
    use rustls::pki_types::PrivateKeyDer;
    use rustls::{ServerConfig, ServerConnection};
    use std::net::TcpListener;

    /// Client config trusting only `cert`.
    pub fn client_config_trusting(cert: &CertificateDer<'static>) -> Arc<ClientConfig> {
        let mut roots = RootCertStore::empty();
        roots.add(cert.clone()).unwrap();
        Arc::new(
            ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
                .with_safe_default_protocol_versions()
                .unwrap()
                .with_root_certificates(roots)
                .with_no_client_auth(),
        )
    }

    /// Serve `responses.len()` TLS connections on 127.0.0.1, one
    /// canned response each; the request text is sent back over
    /// the returned channel. Returns (port, server cert, requests).
    pub fn serve_tls(
        responses: Vec<Vec<u8>>,
    ) -> (
        u16,
        CertificateDer<'static>,
        std::sync::mpsc::Receiver<String>,
    ) {
        let key = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
        let cert_der = CertificateDer::from(key.cert.der().to_vec());
        let key_der = PrivateKeyDer::try_from(key.signing_key.serialize_der()).unwrap();
        let server_cfg = Arc::new(
            ServerConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
                .with_safe_default_protocol_versions()
                .unwrap()
                .with_no_client_auth()
                .with_single_cert(vec![cert_der.clone()], key_der)
                .unwrap(),
        );
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for response in responses {
                let Ok((tcp, _)) = listener.accept() else {
                    return;
                };
                let conn = ServerConnection::new(server_cfg.clone()).unwrap();
                let mut tls = StreamOwned::new(conn, tcp);
                let mut req = Vec::new();
                let mut buf = [0u8; 4096];
                // Read the head, then any Content-Length body.
                while let Ok(n @ 1..) = tls.read(&mut buf) {
                    req.extend_from_slice(&buf[..n]);
                    if let Some(end) = req.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&req[..end]).to_ascii_lowercase();
                        let len: usize = head
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:"))
                            .and_then(|v| v.trim().parse().ok())
                            .unwrap_or(0);
                        if req.len() >= end + 4 + len {
                            break;
                        }
                    }
                }
                let _ = tx.send(String::from_utf8_lossy(&req).into_owned());
                let _ = tls.write_all(&response);
                let _ = tls.flush();
                tls.conn.send_close_notify();
                let _ = tls.conn.complete_io(&mut tls.sock);
            }
        });
        (port, cert_der, rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_url_defaults_ports_and_rejects_bad_input() {
        let u = parse_url("https://api.example.com/v1").unwrap();
        assert_eq!(
            (u.tls, u.host.as_str(), u.port, u.path.as_str()),
            (true, "api.example.com", 443, "/v1")
        );
        let u = parse_url("http://[::1]:8080").unwrap();
        assert_eq!(
            (u.host.as_str(), u.port, u.authority.as_str()),
            ("::1", 8080, "[::1]:8080")
        );
        assert_eq!(u.path, "");
        assert_eq!(parse_url("http://h").unwrap().port, 80);
        assert_eq!(parse_url("https://h?x=1#f").unwrap().path, "/?x=1");
        assert!(parse_url("ftp://h").is_err());
        assert!(parse_url("https://user:pw@h/").is_err());
        assert!(parse_url("https://h:notaport/").is_err());
        assert!(parse_url("https:///path").is_err());
        assert!(parse_url("https://h/\r\nX: y").is_err());
    }

    #[test]
    fn resolve_location_forms() {
        let base = parse_url("https://hub.example/a/b").unwrap();
        assert_eq!(
            resolve_location(&base, "/c?d").unwrap(),
            parse_url("https://hub.example/c?d").unwrap()
        );
        assert_eq!(
            resolve_location(&base, "//cdn.example/x").unwrap(),
            parse_url("https://cdn.example/x").unwrap()
        );
        assert!(!resolve_location(&base, "http://plain/x").unwrap().tls);
        let rel = |loc: &str| resolve_location(&base, loc).unwrap().path;
        assert_eq!(rel("c"), "/a/c");
        assert_eq!(rel("./c/d?x=1#frag"), "/a/c/d?x=1");
        assert_eq!(rel("../c"), "/c");
        assert_eq!(rel("../../../c"), "/c");
        assert_eq!(rel("?q=2"), "/a/b?q=2");
        assert_eq!(rel("."), "/a/");
        let root = parse_url("https://hub.example").unwrap();
        assert_eq!(resolve_location(&root, "x").unwrap().path, "/x");
        assert!(resolve_location(&base, "").is_err());
        assert!(resolve_location(&base, "#only").is_err());
    }

    #[test]
    fn chunked_reader_decodes_and_detects_truncation() {
        let body: &[u8] = b"4;ext=1\r\nWiki\r\n6\r\npedia \r\nE\r\nin \r\n\r\nchunks.\r\n0\r\nX-Trailer: y\r\n\r\nleftover";
        let mut out = Vec::new();
        ChunkedReader::new(body).read_to_end(&mut out).unwrap();
        assert_eq!(out, b"Wikipedia in \r\n\r\nchunks.");
        // Closed before the zero-size chunk: an error, not EOF.
        let mut out = Vec::new();
        let err = ChunkedReader::new(&b"4\r\nWiki\r\n"[..])
            .read_to_end(&mut out)
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
        let err = ChunkedReader::new(&b"zz\r\n"[..])
            .read_to_end(&mut Vec::new())
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn tls_round_trip_and_untrusted_certificate_is_refused() {
        let (port, cert, rx) = test_support::serve_tls(vec![
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"
                    .to_vec();
                2
            ]);
        let url = parse_url(&format!("https://localhost:{port}/")).unwrap();
        // Default roots do not trust a self-signed certificate.
        let err = match connect(&url, Duration::from_secs(5), Duration::from_secs(5)) {
            Ok(_) => panic!("self-signed certificate was accepted"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("TLS handshake"), "{err}");
        TEST_TLS_CONFIG
            .with(|c| *c.borrow_mut() = Some(test_support::client_config_trusting(&cert)));
        let mut conn = connect(&url, Duration::from_secs(5), Duration::from_secs(5)).unwrap();
        conn.write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .unwrap();
        let mut out = String::new();
        conn.read_to_string(&mut out).unwrap();
        assert!(out.ends_with("\r\n\r\nok"), "{out}");
        // The first served connection was the refused handshake.
        assert_eq!(rx.recv().unwrap(), "");
        assert!(rx.recv().unwrap().starts_with("GET / HTTP/1.1"));
        TEST_TLS_CONFIG.with(|c| *c.borrow_mut() = None);
    }
}
