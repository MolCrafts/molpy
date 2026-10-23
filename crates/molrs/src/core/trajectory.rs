//! Backend-agnostic trajectory + observable logical model: [`Trajectory`] and [`ObservableRecord`].

use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::core::Column;
use crate::core::Frame;
use crate::core::MolRsError;
use crate::op::F;

/// Trajectory-like list of frame states plus shared indexing arrays.
///
/// A plain frame-sequence carrier (frames plus optional step/time index
/// arrays), not a record aggregate — the canonical entity is
/// [`Frame`](crate::core::Frame). [`Trajectory::default`] is the empty
/// trajectory.
#[derive(Debug, Clone, Default)]
pub struct Trajectory {
    /// Ordered frame-like states.
    pub frames: Vec<Frame>,
    /// Optional discrete step indices.
    pub step: Option<Vec<i64>>,
    /// Optional physical time values.
    pub time: Option<Vec<F>>,
}

impl Trajectory {
    /// Build a trajectory from frame states.
    pub fn from_frames(frames: Vec<Frame>) -> Self {
        Self {
            frames,
            step: None,
            time: None,
        }
    }

    /// Number of states.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns true when no states are stored.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// The sub-trajectory of the states at `indices`, in that order, with
    /// their `step` / `time` labels.
    ///
    /// # Panics
    ///
    /// If an index is out of range, or a present `step` / `time` axis is
    /// shorter than the frames (see [`validate`](Self::validate)).
    pub fn select(&self, indices: &[usize]) -> Self {
        Self {
            frames: indices.iter().map(|&i| self.frames[i].clone()).collect(),
            step: self
                .step
                .as_ref()
                .map(|step| indices.iter().map(|&i| step[i]).collect()),
            time: self
                .time
                .as_ref()
                .map(|time| indices.iter().map(|&i| time[i]).collect()),
        }
    }

    /// Validate shared axis lengths.
    pub fn validate(&self) -> Result<(), MolRsError> {
        let n = self.frames.len();
        if let Some(step) = &self.step
            && step.len() != n
        {
            return Err(MolRsError::validation(format!(
                "trajectory.step length mismatch: expected {}, got {}",
                n,
                step.len()
            )));
        }
        if let Some(time) = &self.time
            && time.len() != n
        {
            return Err(MolRsError::validation(format!(
                "trajectory.time length mismatch: expected {}, got {}",
                n,
                time.len()
            )));
        }
        Ok(())
    }
}

/// Observable kind aligned with the observable metadata contract.
///
/// `scalar` and `vector` are the contract's kinds. A producer may declare
/// another one (in a `meta/modules` module); this build carries such a kind
/// verbatim as [`Other`](Self::Other) and writes it back unchanged rather than
/// refusing the record. Serialized as its contract spelling, a plain string.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum ObservableKind {
    /// One value per sample.
    Scalar,
    /// An ordered tuple of components per sample.
    Vector,
    /// A kind this build does not define, kept as written.
    Other(String),
}

impl ObservableKind {
    /// Contract spelling of the kind, as written to `observables/meta/<name>/kind`.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Scalar => "scalar",
            Self::Vector => "vector",
            Self::Other(kind) => kind,
        }
    }
}

impl From<&str> for ObservableKind {
    /// Parse the contract spelling. A spelling this build does not define is
    /// [`Other`](Self::Other), never an error.
    fn from(kind: &str) -> Self {
        match kind {
            "scalar" => Self::Scalar,
            "vector" => Self::Vector,
            other => Self::Other(other.to_string()),
        }
    }
}

impl From<String> for ObservableKind {
    fn from(kind: String) -> Self {
        match kind.as_str() {
            "scalar" => Self::Scalar,
            "vector" => Self::Vector,
            _ => Self::Other(kind),
        }
    }
}

impl From<ObservableKind> for String {
    fn from(kind: ObservableKind) -> Self {
        match kind {
            ObservableKind::Other(kind) => kind,
            known => known.as_str().to_string(),
        }
    }
}

impl std::fmt::Display for ObservableKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Raw observable payload.
#[derive(Debug, Clone)]
pub enum ObservableValues {
    /// Typed ndarray-style data.
    Column(Column),
}

/// Named observable with semantic metadata: a named, typed observable
/// payload, not a record aggregate.
#[derive(Debug, Clone)]
pub struct ObservableRecord {
    pub name: String,
    pub kind: ObservableKind,
    pub description: String,
    pub time_dependent: bool,
    pub unit: Option<String>,
    pub axes: Vec<String>,
    pub sampling: Option<String>,
    pub domain: Option<String>,
    pub target: Option<String>,
    pub extra: JsonMap<String, JsonValue>,
    pub values: ObservableValues,
}

impl ObservableRecord {
    /// Build an observable of `kind` named `name` carrying `values`, with no
    /// unit, axes, sampling, domain, target or description (set those fields
    /// directly).
    pub fn new(name: impl Into<String>, kind: ObservableKind, values: Column) -> Self {
        Self {
            name: name.into(),
            kind,
            description: String::new(),
            time_dependent: false,
            unit: None,
            axes: Vec::new(),
            sampling: None,
            domain: None,
            target: None,
            extra: JsonMap::new(),
            values: ObservableValues::Column(values),
        }
    }

    /// The metadata document as the `observables/meta/<name>` attribute map
    /// stores it: the named keys that are set, then every producer key.
    pub fn document(&self) -> JsonMap<String, JsonValue> {
        let mut attrs = self.extra.clone();
        attrs.insert("kind".into(), self.kind.as_str().into());
        attrs.insert("description".into(), self.description.clone().into());
        attrs.insert("time_dependent".into(), self.time_dependent.into());
        if let Some(unit) = &self.unit {
            attrs.insert("unit".into(), unit.clone().into());
        }
        if !self.axes.is_empty() {
            attrs.insert("axes".into(), self.axes.clone().into());
        }
        if let Some(sampling) = &self.sampling {
            attrs.insert("sampling".into(), sampling.clone().into());
        }
        if let Some(domain) = &self.domain {
            attrs.insert("domain".into(), domain.clone().into());
        }
        if let Some(target) = &self.target {
            attrs.insert("target".into(), target.clone().into());
        }
        attrs
    }

    /// The observable `name` from its metadata document — the inverse of
    /// [`document`](Self::document) — and its `values`. A key
    /// [`document`](Self::document) does not name is a producer key, kept in
    /// [`extra`](Self::extra); a kind this build does not define is carried
    /// as [`ObservableKind::Other`]. An absent `description` is empty and an
    /// absent `time_dependent` false; the other named keys are optional.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] naming the key for a missing `kind`, or a
    /// named key of the wrong JSON type (an `axes` entry included) — never
    /// read as its default.
    pub fn from_document(
        name: &str,
        mut attrs: JsonMap<String, JsonValue>,
        values: Column,
    ) -> Result<Self, MolRsError> {
        let mut take = |key: &str| attrs.remove(key);
        let wrong = |key: &str, what: &str, value: &JsonValue| {
            MolRsError::validation(format!(
                "observable '{name}': '{key}' must be {what}, not {value}"
            ))
        };
        let text = |key: &str, value: Option<JsonValue>| -> Result<Option<String>, MolRsError> {
            match value {
                None => Ok(None),
                Some(JsonValue::String(s)) => Ok(Some(s)),
                Some(other) => Err(wrong(key, "a string", &other)),
            }
        };
        let kind = text("kind", take("kind"))?
            .map(ObservableKind::from)
            .ok_or_else(|| MolRsError::validation(format!("observable '{name}' has no 'kind'")))?;
        let description = text("description", take("description"))?.unwrap_or_default();
        let time_dependent = match take("time_dependent") {
            None => false,
            Some(JsonValue::Bool(b)) => b,
            Some(other) => return Err(wrong("time_dependent", "a bool", &other)),
        };
        let unit = text("unit", take("unit"))?;
        let axes = match take("axes") {
            None => Vec::new(),
            Some(JsonValue::Array(items)) => items
                .into_iter()
                .map(|item| match item {
                    JsonValue::String(s) => Ok(s),
                    other => Err(wrong("axes", "a list of strings", &other)),
                })
                .collect::<Result<_, _>>()?,
            Some(other) => return Err(wrong("axes", "a list of strings", &other)),
        };
        let sampling = text("sampling", take("sampling"))?;
        let domain = text("domain", take("domain"))?;
        let target = text("target", take("target"))?;
        Ok(Self {
            name: name.to_string(),
            kind,
            description,
            time_dependent,
            unit,
            axes,
            sampling,
            domain,
            target,
            extra: attrs,
            values: ObservableValues::Column(values),
        })
    }

    /// Validate the observable payload against the declared kind.
    ///
    /// Every kind, an [`ObservableKind::Other`] included, is carried as one
    /// column; the contract fixes no shape per kind.
    pub fn validate(&self) -> Result<(), MolRsError> {
        match &self.values {
            ObservableValues::Column(_) => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `from_document` reads back what `document` writes, keeps producer
    /// keys, and refuses a named key of the wrong type instead of defaulting.
    #[test]
    fn an_observable_document_round_trips_and_wrong_types_are_refused() {
        let column = || Column::from(ndarray::Array1::from_vec(vec![1.0_f64, 2.0]).into_dyn());
        let mut obs = ObservableRecord::new("rg", ObservableKind::Scalar, column());
        obs.unit = Some("angstrom".into());
        obs.axes = vec!["time".into()];
        obs.time_dependent = true;
        obs.extra.insert("producer".into(), "molpy".into());
        let back = ObservableRecord::from_document("rg", obs.document(), column()).unwrap();
        assert_eq!(back.document(), obs.document());
        assert_eq!(back.extra.get("producer"), Some(&JsonValue::from("molpy")));

        for (key, value) in [
            ("time_dependent", JsonValue::from("yes")),
            ("axes", serde_json::json!(["time", 3])),
            ("unit", JsonValue::from(1)),
        ] {
            let mut doc = obs.document();
            doc.insert(key.into(), value);
            let err = ObservableRecord::from_document("rg", doc, column()).unwrap_err();
            assert!(err.to_string().contains(key), "{err}");
        }
        let mut doc = obs.document();
        doc.remove("kind");
        assert!(ObservableRecord::from_document("rg", doc, column()).is_err());
    }

    #[test]
    fn empty_trajectory_has_no_frames() {
        let traj = Trajectory::default();
        assert_eq!(traj.len(), 0);
        assert!(traj.is_empty());
    }

    #[test]
    fn from_frames_counts_states() {
        let traj = Trajectory::from_frames(vec![Frame::new(), Frame::new()]);
        assert_eq!(traj.len(), 2);
        assert!(!traj.is_empty());
        traj.validate().unwrap();
    }

    #[test]
    fn observable_kind_parses_known_and_keeps_unknown_spellings() {
        assert_eq!(ObservableKind::from("scalar"), ObservableKind::Scalar);
        assert_eq!(ObservableKind::from("vector"), ObservableKind::Vector);
        let other = ObservableKind::from("spectrum");
        assert_eq!(other, ObservableKind::Other("spectrum".into()));
        assert_eq!(other.as_str(), "spectrum");
        assert_eq!(String::from(other.clone()), "spectrum");
        assert_eq!(
            ObservableKind::from(String::from("scalar")),
            ObservableKind::Scalar
        );
        assert_eq!(
            serde_json::to_value(&other).unwrap(),
            serde_json::json!("spectrum")
        );
        assert_eq!(
            serde_json::from_value::<ObservableKind>(serde_json::json!("vector")).unwrap(),
            ObservableKind::Vector
        );
    }

    #[test]
    fn select_keeps_the_chosen_states_and_their_labels() {
        let frames = (0..4)
            .map(|i| {
                let mut f = Frame::new();
                f.meta.insert("i", crate::core::MetaValue::I64(i));
                f
            })
            .collect();
        let mut traj = Trajectory::from_frames(frames);
        traj.step = Some(vec![0, 10, 20, 30]);
        traj.time = Some(vec![0.0, 0.5, 1.0, 1.5]);

        let sub = traj.select(&[3, 1]);
        assert_eq!(sub.len(), 2);
        assert_eq!(sub.step, Some(vec![30, 10]));
        assert_eq!(sub.time, Some(vec![1.5, 0.5]));
        assert_eq!(
            sub.frames[0].meta.get("i"),
            Some(&crate::core::MetaValue::I64(3))
        );
        sub.validate().unwrap();

        let unlabeled = Trajectory::from_frames(vec![Frame::new(); 2]).select(&[1]);
        assert_eq!(
            (unlabeled.len(), unlabeled.step, unlabeled.time),
            (1, None, None)
        );
    }

    #[test]
    fn observable_record_new_carries_the_stated_kind() {
        let values = Column::from(ndarray::arr1(&[1.0 as F, 2.0]).into_dyn());
        let rec = ObservableRecord::new("spectrum", ObservableKind::from("spectrum"), values);
        assert_eq!(rec.name, "spectrum");
        assert_eq!(rec.kind, ObservableKind::Other("spectrum".into()));
        assert!(rec.unit.is_none() && rec.axes.is_empty() && !rec.time_dependent);
        rec.validate().unwrap();
    }

    #[test]
    fn mismatched_step_axis_fails_validation() {
        let mut traj = Trajectory::from_frames(vec![Frame::new(), Frame::new()]);
        traj.step = Some(vec![0]); // length 1 != 2 frames
        assert!(traj.validate().is_err());
    }
}
