//! Force-directed graph layout, computed off the UI thread.
//! Positions are cached per graph revision; a worker thread runs
//! the simulation and posts the result back over a channel so the
//! egui frame loop never blocks on layout.

use gridtalk_graph::{Graph, NodeKind};
use std::collections::BTreeMap;
use std::sync::mpsc::{Receiver, Sender};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pos {
    pub x: f32,
    pub y: f32,
}

/// Deterministic force layout: nodes start on a circle by kind
/// (workbook centre, sheets inner ring, regions/columns outer),
/// then repel/attract iterations settle them. Pure function of the
/// graph, so it is unit-testable without a display.
pub fn compute_layout(
    graph: &Graph,
    width: f32,
    height: f32,
    iterations: usize,
) -> BTreeMap<usize, Pos> {
    let cx = width / 2.0;
    let cy = height / 2.0;
    let mut pos: BTreeMap<usize, Pos> = BTreeMap::new();
    for (i, n) in graph.nodes.iter().enumerate() {
        let (radius, phase) = match n.kind {
            NodeKind::Workbook => (0.0, 0.0),
            NodeKind::Sheet => (70.0, i as f32),
            NodeKind::Region => (150.0, i as f32 * 1.7),
            NodeKind::Column => (230.0, i as f32 * 2.3),
        };
        let angle = phase * 2.39996; // golden angle spread
        pos.insert(
            n.id,
            Pos {
                x: cx + radius * angle.cos(),
                y: cy + radius * angle.sin(),
            },
        );
    }
    let ids: Vec<usize> = graph.nodes.iter().map(|n| n.id).collect();
    for _ in 0..iterations {
        // Repulsion between every pair (graphs here are lifted and small).
        let mut force: BTreeMap<usize, (f32, f32)> =
            ids.iter().map(|id| (*id, (0.0, 0.0))).collect();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let (a, b) = (pos[&ids[i]], pos[&ids[j]]);
                let (dx, dy) = (a.x - b.x, a.y - b.y);
                let d2 = (dx * dx + dy * dy).max(25.0);
                let f = 2600.0 / d2;
                let (fx, fy) = (dx * f, dy * f);
                let fa = force.get_mut(&ids[i]).unwrap();
                fa.0 += fx;
                fa.1 += fy;
                let fb = force.get_mut(&ids[j]).unwrap();
                fb.0 -= fx;
                fb.1 -= fy;
            }
        }
        // Attraction along edges.
        for e in &graph.edges {
            if let (Some(a), Some(b)) = (pos.get(&e.from).copied(), pos.get(&e.to).copied()) {
                let (dx, dy) = (b.x - a.x, b.y - a.y);
                let fa = force.get_mut(&e.from).unwrap();
                fa.0 += dx * 0.02;
                fa.1 += dy * 0.02;
                let fb = force.get_mut(&e.to).unwrap();
                fb.0 -= dx * 0.02;
                fb.1 -= dy * 0.02;
            }
        }
        for id in &ids {
            let (fx, fy) = force[id];
            let p = pos.get_mut(id).unwrap();
            p.x = (p.x + fx.clamp(-12.0, 12.0)).clamp(18.0, width - 18.0);
            p.y = (p.y + fy.clamp(-12.0, 12.0)).clamp(18.0, height - 18.0);
        }
    }
    pos
}

/// Handle to a background layout computation.
pub struct LayoutWorker {
    pub revision: u64,
    pub rx: Receiver<(u64, BTreeMap<usize, Pos>)>,
}

pub fn spawn_layout(graph: Graph, revision: u64, width: f32, height: f32) -> LayoutWorker {
    let (tx, rx): (Sender<(u64, BTreeMap<usize, Pos>)>, _) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let positions = compute_layout(&graph, width, height, 140);
        let _ = tx.send((revision, positions));
    });
    LayoutWorker { revision, rx }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gridtalk_graph::{build_graph, parse_delimited};

    #[test]
    fn layout_covers_every_node_inside_bounds() {
        let wb = parse_delimited("d", "Sheet1", "A,B\n1,2\n", ',');
        let (g, _) = build_graph(&wb);
        let pos = compute_layout(&g, 640.0, 480.0, 60);
        assert_eq!(pos.len(), g.nodes.len());
        for p in pos.values() {
            assert!(p.x >= 18.0 && p.x <= 622.0 && p.y >= 18.0 && p.y <= 462.0);
        }
    }
}
