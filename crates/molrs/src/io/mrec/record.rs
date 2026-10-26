//! MolRec record aggregate — L2 of the MolRec contract.

use std::collections::BTreeMap;

use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::core::Frame;
use crate::core::MolRsError;
use crate::core::{ObservableRecord, Trajectory};
use crate::io::mrec::ForceFieldSection;

/// Named observables of a record, keyed by observable name.
///
/// Data and semantic metadata are one unit: the contract forbids standalone
/// observable arrays, so a record can only carry a fully described
/// [`ObservableRecord`].
#[derive(Debug, Clone, Default)]
pub struct Observables {
    records: BTreeMap<String, ObservableRecord>,
}

impl Observables {
    /// Create an empty collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of observables.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns true when no observable is stored.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Returns true when `name` is present.
    pub fn contains(&self, name: &str) -> bool {
        self.records.contains_key(name)
    }

    /// Borrow one observable.
    pub fn get(&self, name: &str) -> Option<&ObservableRecord> {
        self.records.get(name)
    }

    /// Insert an observable, replacing any earlier one of the same name.
    pub fn insert(&mut self, record: ObservableRecord) -> Result<(), MolRsError> {
        record.validate()?;
        self.records.insert(record.name.clone(), record);
        Ok(())
    }

    /// Remove one observable.
    pub fn remove(&mut self, name: &str) -> Option<ObservableRecord> {
        self.records.remove(name)
    }

    /// Observable names, sorted.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.records.keys()
    }

    /// Iterate over `(name, record)` pairs, sorted by name.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ObservableRecord)> {
        self.records.iter()
    }
}

/// One self-describing record: the unit of interchange between MolCrafts tools.
///
/// Sections map one-to-one onto the contract's root layout. `meta` is always
/// written; the remaining sections are optional. A root section the reader does
/// not interpret is ignored, never reinterpreted.
///
/// It is backend-neutral: the in-memory aggregate, not a file format. Reading
/// and writing a record as a `*.mrec` directory is [`crate::io::mrec`]'s
/// (feature `zarr`).
///
/// Contract: <https://github.com/MolCrafts/molrec> (`docs/spec/overview.md`).
/// The scientific path brand is the `*.mrec/` suffix.
#[derive(Debug, Clone, Default)]
pub struct MolRec {
    /// Record-level metadata, the producer's own document. Keys a reader does
    /// not recognise are kept.
    pub meta: JsonMap<String, JsonValue>,
    /// How the record was produced (run surface).
    pub method: JsonMap<String, JsonValue>,
    /// Lifecycle / progress (run surface).
    pub status: JsonMap<String, JsonValue>,
    /// Append-only run measurements (run surface): the catalog / summary
    /// document, stored as `metrics/` group attributes.
    pub metrics: JsonMap<String, JsonValue>,
    /// Closed metric curves, keyed by the logical series name (`train/loss`).
    /// Each series is one float64 array at `metrics/series/<safe name>`.
    pub metrics_series: BTreeMap<String, Vec<f64>>,
    /// The point's step (`i64`) for each series that has steps, aligned index
    /// for index with [`metrics_series`](Self::metrics_series). A series with
    /// no steps has no entry. Stored at `metrics/steps/<safe name>`.
    pub metrics_steps: BTreeMap<String, Vec<i64>>,
    /// The point's RFC 3339 time for each series that has one, aligned with
    /// the series. Stored at `metrics/wall_time/<safe name>`.
    pub metrics_wall_time: BTreeMap<String, Vec<String>>,
    /// The live JSONL WAL (`metrics/metrics.jsonl`) as stored, when the
    /// record carries one. The doors keep the bytes; they do not densify.
    pub metrics_wal: Option<String>,
    /// System definition — topology and types, without instantaneous state.
    pub system: Option<Frame>,
    /// Instantaneous snapshot.
    pub frame: Option<Frame>,
    /// Ordered frame sequence.
    pub trajectory: Option<Trajectory>,
    /// The force field the record's types link into, as data: the document
    /// and one table per style (molrec `docs/spec/forcefield.md`).
    pub forcefield: Option<ForceFieldSection>,
    /// Named scientific results.
    pub observables: Observables,
}

impl MolRec {
    /// Create an empty record.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a frame to the trajectory section, creating it when absent.
    pub fn add_frame(&mut self, frame: Frame) {
        self.trajectory
            .get_or_insert_with(Trajectory::default)
            .frames
            .push(frame);
    }

    /// Number of frames the record carries: the trajectory length when present,
    /// otherwise one for a bare snapshot.
    pub fn n_frames(&self) -> usize {
        match &self.trajectory {
            Some(traj) => traj.len(),
            None => usize::from(self.frame.is_some()),
        }
    }

    /// Check the contract's minimum record shape.
    ///
    /// A record must carry at least one of `frame`, `system`, `trajectory`,
    /// `forcefield`, or `status`; a Run-shaped record (`meta` + `status`) needs
    /// no frame, a trajectory is a state section in its own right — a sequence
    /// of frames stands alone, without a snapshot beside it — and `meta` +
    /// `forcefield` is a force-field package.
    ///
    /// Shape only. This does not check that the sections agree with the Frame
    /// schema — that is `crate::core::schema::Validator`'s job, and the read
    /// and write doors run it separately.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Validation`] when all five of those sections are absent
    /// (`status` counts as absent when it is empty), or whatever
    /// [`ForceFieldSection::validate`] reports about the force field. The check is transitive,
    /// so it also returns whatever [`Trajectory::validate`] reports — a `step`
    /// or `time` axis whose length does not match the frame count, in a message
    /// naming that axis — and whatever each stored [`ObservableRecord`] reports
    /// about itself, which in this build is nothing: every kind-and-data
    /// pairing the type can hold is valid. The first failure wins.
    pub fn validate(&self) -> Result<(), MolRsError> {
        if self.frame.is_none()
            && self.system.is_none()
            && self.trajectory.is_none()
            && self.forcefield.is_none()
            && self.status.is_empty()
        {
            return Err(MolRsError::validation(
                "record must carry at least one of 'frame', 'system', 'trajectory', \
                 'forcefield', or 'status'",
            ));
        }
        if let Some(forcefield) = &self.forcefield {
            forcefield.validate()?;
        }
        if let Some(traj) = &self.trajectory {
            traj.validate()?;
        }
        for record in self.observables.records.values() {
            record.validate()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Column;
    use molrs::core::ObservableKind;
    use ndarray::ArrayD;

    fn scalar_column(values: &[f64]) -> Column {
        Column::from(ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap())
    }

    #[test]
    fn empty_record_fails_minimum_shape() {
        assert!(MolRec::new().validate().is_err());
    }

    #[test]
    fn frame_only_record_is_valid() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        rec.validate().unwrap();
    }

    #[test]
    fn system_only_record_is_valid() {
        let mut rec = MolRec::new();
        rec.system = Some(Frame::new());
        rec.validate().unwrap();
    }

    /// A trajectory is a state section like any other: a record that carries
    /// `meta` plus a sequence of frames — and no snapshot, no system, no
    /// status — is a complete record, not a defective one.
    ///
    /// Contract: `../molrec/docs/spec/overview.md` lists `trajectory`
    /// alongside `frame`, `system` and `status`, so a trajectory-only
    /// record carries no duplicate of frame 0 in `frame`.
    #[test]
    fn a_trajectory_only_record_validates() {
        let mut rec = MolRec::new();
        rec.meta.insert("creator".into(), "unit-test".into());
        rec.add_frame(Frame::new());
        rec.add_frame(Frame::new());

        assert!(rec.frame.is_none(), "no snapshot section");
        assert!(rec.system.is_none(), "no system section");
        assert!(rec.status.is_empty(), "no status section");
        rec.validate()
            .expect("a record whose only state section is a trajectory is valid");
    }

    #[test]
    fn a_forcefield_only_record_validates_and_its_forcefield_is_checked() {
        let mut rec = MolRec::new();
        let mut ff = ForceFieldSection::default();
        ff.document.insert("name".into(), "pkg".into());
        rec.forcefield = Some(ff);
        assert!(
            rec.validate().is_err(),
            "a document with no units is refused"
        );
        rec.forcefield
            .as_mut()
            .unwrap()
            .document
            .insert("units".into(), serde_json::json!({"preset": "real"}));
        rec.validate()
            .expect("meta + forcefield is a force-field package");
    }

    #[test]
    fn run_shaped_record_needs_no_frame() {
        let mut rec = MolRec::new();
        rec.status.insert("state".into(), "running".into());
        rec.validate().unwrap();
    }

    #[test]
    fn bare_snapshot_counts_one_frame() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        assert_eq!(rec.n_frames(), 1);
    }

    #[test]
    fn trajectory_length_overrides_snapshot_count() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        rec.add_frame(Frame::new());
        rec.add_frame(Frame::new());
        assert_eq!(rec.n_frames(), 2);
    }

    #[test]
    fn empty_record_counts_no_frames() {
        assert_eq!(MolRec::new().n_frames(), 0);
    }

    #[test]
    fn mismatched_trajectory_axis_fails_record_validation() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        rec.add_frame(Frame::new());
        rec.trajectory.as_mut().unwrap().step = Some(vec![0, 1]); // 2 steps, 1 frame
        assert!(rec.validate().is_err());
    }

    #[test]
    fn observables_round_trip_by_name() {
        let mut obs = Observables::new();
        obs.insert(ObservableRecord::new(
            "total_energy",
            ObservableKind::Scalar,
            scalar_column(&[1.0, 2.0]),
        ))
        .unwrap();
        assert!(obs.contains("total_energy"));
        assert_eq!(obs.len(), 1);
        assert_eq!(obs.get("total_energy").unwrap().name, "total_energy");
    }

    #[test]
    fn inserting_same_observable_name_replaces() {
        let mut obs = Observables::new();
        obs.insert(ObservableRecord::new(
            "e",
            ObservableKind::Scalar,
            scalar_column(&[1.0]),
        ))
        .unwrap();
        obs.insert(ObservableRecord::new(
            "e",
            ObservableKind::Vector,
            scalar_column(&[2.0]),
        ))
        .unwrap();
        assert_eq!(obs.len(), 1);
    }
}
