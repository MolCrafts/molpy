use super::*;

// =======================================================================
// I. Reader — one cursor, two access shapes
// =======================================================================

/// `FrameIterator` over a `MrecReader` yields exactly what a `frame(i)`
/// loop yields (ac-014).
///
/// This is the payoff of dropping the `Reader` supertrait from
/// `TrajectoryReader`: the existing generic consumers work against a
/// store-backed sequence with no `BufRead` anywhere.
#[test]
fn the_frame_iterator_yields_what_frame_yields() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let frames = ragged_frames();
    write_all(&store, &frames);

    let mut iterated = open_sequence(&store);
    let streamed: Vec<Vec<f64>> = iterated
        .iter()
        .map(|frame| atoms_x(&frame.unwrap()))
        .collect();

    let mut indexed = open_sequence(&store);
    let random: Vec<Vec<f64>> = (0..frames.len() as u64)
        .map(|index| atoms_x(&frame_at(&mut indexed, index)))
        .collect();

    assert_eq!(streamed, random, "the iterator is the frame(i) loop");
    assert_eq!(
        streamed,
        frames.iter().map(atoms_x).collect::<Vec<Vec<f64>>>(),
        "and both are what was appended"
    );
}

/// `to_trajectory()` is the named lazy-to-eager conversion: the same
/// frames, materialized.
#[test]
fn to_trajectory_carries_the_same_frames_as_frame() {
    let dir = TempDir::new().unwrap();
    let store = storage_in(&dir);
    let frames = ragged_frames();
    write_all(&store, &frames);

    let seq = open_sequence(&store);
    let eager = seq.to_trajectory().unwrap();

    assert_eq!(eager.frames.len(), frames.len());
    assert_eq!(
        eager.frames.iter().map(atoms_x).collect::<Vec<Vec<f64>>>(),
        frames.iter().map(atoms_x).collect::<Vec<Vec<f64>>>(),
        "the eager carrier holds the same bits as the lazy cursor"
    );
}
