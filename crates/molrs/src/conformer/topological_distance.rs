//! Topological (bond-count) distances between atoms, for the
//! distance-geometry bounds.

use std::collections::VecDeque;

/// All-pairs topological distance matrix (number of bonds between atoms) over a
/// plain adjacency list, computed by a BFS from each node. Unreachable pairs
/// are left as [`usize::MAX`].
pub(crate) fn topological_distances(adjacency: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adjacency.len();
    let mut dist = vec![vec![usize::MAX; n]; n];
    for start in 0..n {
        let row = &mut dist[start];
        row[start] = 0;
        let mut queue = VecDeque::from([start]);
        while let Some(i) = queue.pop_front() {
            let d = row[i];
            for &j in &adjacency[i] {
                if row[j] == usize::MAX {
                    row[j] = d + 1;
                    queue.push_back(j);
                }
            }
        }
    }
    dist
}
