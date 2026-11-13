//! The part of GridTalk that answers a question about a workbook:
//! the typed tools (`tools`), the deterministic planner that reads
//! what a question names (`investigate`), and the loop in which a
//! model chooses its own tool calls (`agent`). The CLI, the MCP
//! server, bench and the desktop UI all call into this crate, so a
//! tool or the loop is changed here and nowhere else.

pub mod agent;
pub mod investigate;
pub mod tools;
