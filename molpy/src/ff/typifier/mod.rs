//! Python bindings for `molrs::ff::typifier` (`molpy.ff.typifier`): the
//! subclassable [`PyTypifier`] base and the [`PyTypeAssignment`] its ``assign`` hook
//! returns, the native typifiers — MMFF94 / MMFF94s, OPLS-AA, element, and
//! the antechamber-derived [`atd`] and [`gaff`] — and `assign_cmaps`, which
//! types a molecule's CMAP crossterms from its dihedrals.
//!
//! A typifier's contract is `typify`: compiling potentials from what it
//! defined is `molpy.ff.compile.PotentialCompiler`'s. There is deliberately
//! no one-step `build(mol)`: one such shortcut once silently omitted the whole
//! electrostatic term because no `ForceField` defined `pair/mmff_ele`.
//!
//! # References
//!
//! - Halgren, T.A. (1996). J. Comput. Chem. 17, 490-519. (MMFF94 force field)
//! - Halgren, T.A. (1999). J. Comput. Chem. 20, 720-729. (MMFF94s option)

pub mod atd;
pub mod gaff;

use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyMapping, PySuper, PyTuple, PyType};

use molrs::ff::forcefield::ForceField;
use molrs::ff::typifier::ElementTypifier;
use molrs::ff::typifier::OplsAaTypifier;
use molrs::ff::typifier::mmff::{Mmff94Typifier, Mmff94sTypifier};
use molrs::ff::typifier::{Annotation, TypeAssignment, Typifier, Typing};

use crate::core::graph_views::RelationClass;
use crate::core::molgraph::{PyAtomistic, py_to_prop};
use crate::ff::forcefield::{PyForceField, params_from_dict};
use crate::io::PySource;

/// Where a [`PyTypifier`]'s typing state lives.
enum TypifierState {
    /// A native typifier class: the Rust base owns the matcher and the output.
    Native(Typing<Box<dyn Typifier + Send + Sync>>),
    /// A Python subclass: the matcher is its ``assign`` method and this base
    /// holds the output, unset until first seeded (see [`PyTypifier::seed`]).
    Python(Option<ForceField>),
}

fn unseeded_output() -> PyErr {
    PyRuntimeError::new_err("typifier output accessed before it was seeded")
}

/// The base of every graph typifier: one ``assign`` hook plus the output force
/// field its typing accumulates.
///
/// Exposed to Python as ``molpy.ff.typifier.Typifier`` and subclassable. A
/// subclass implements :meth:`assign` (and optionally :meth:`source_forcefield`) and
/// nothing else; :meth:`typify` is the one execution path and the only writer
/// of :meth:`forcefield`. Defining ``typify`` on a subclass raises
/// ``TypeError`` at class creation.
///
/// The native typifier classes (``Mmff94Typifier``, ``Mmff94sTypifier``,
/// ``OplsAaTypifier``, ``AtdTypifier``) extend this base and only construct:
/// their ``assign`` runs the Rust matcher and their ``typify`` the Rust
/// ``Typing::typify``.
#[pyclass(module = "molpy.ff.typifier", name = "Typifier", subclass, generic)]
pub struct PyTypifier {
    state: TypifierState,
}

impl PyTypifier {
    /// The base of a native typifier class: `typifier` wrapped in [`Typing`],
    /// whose output starts as `typifier.source_forcefield().empty_like()`.
    pub(crate) fn native(typifier: impl Typifier + Send + Sync + 'static) -> Self {
        Self {
            state: TypifierState::Native(Typing::new(Box::new(typifier))),
        }
    }

    /// The library's name, for the native classes' `__repr__`; empty for a
    /// Python subclass, whose library is whatever its `source_forcefield()` returns.
    pub(crate) fn source_forcefield_name(&self) -> &str {
        match &self.state {
            TypifierState::Native(typing) => &typing.typifier().source_forcefield().name,
            TypifierState::Python(_) => "",
        }
    }

    /// Seed a Python subclass's output on first access, exactly as
    /// [`Typing::new`] does: `self.source_forcefield().empty_like()` — the library's
    /// name and declared units and special_bonds. A subclass without a
    /// `source_forcefield()` (it raises `NotImplementedError`) gets an empty force field
    /// named after its class. A no-op once seeded, and for a native typifier.
    fn seed(slf: &Bound<'_, Self>) -> PyResult<()> {
        if !matches!(slf.borrow().state, TypifierState::Python(None)) {
            return Ok(());
        }
        let py = slf.py();
        // `source_forcefield()` is dispatched through Python (a subclass overrides it),
        // so no borrow of `slf` is held across the call.
        let seed = match slf.call_method0(intern!(py, "source_forcefield")) {
            Ok(library) => {
                let library = library.cast_into::<PyForceField>().map_err(|err| {
                    PyTypeError::new_err(format!(
                        "source_forcefield() must return a ForceField: {err}"
                    ))
                })?;
                library.borrow().inner.empty_like()
            }
            Err(err) if err.is_instance_of::<PyNotImplementedError>(py) => {
                ForceField::new(&slf.get_type().name()?.to_string())
            }
            Err(err) => return Err(err),
        };
        if let TypifierState::Python(output) = &mut slf.borrow_mut().state {
            // `source_forcefield()` may itself have reached `forcefield()` and seeded.
            output.get_or_insert(seed);
        }
        Ok(())
    }
}

/// A refused typing, as a ``ValueError`` carrying the core's message.
fn typing_err(err: molrs::ff::typifier::TypingError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pymethods]
impl PyTypifier {
    /// A Python-subclass base with an unset output.
    ///
    /// Accepts and ignores any arguments, as ``object.__new__`` does, so a
    /// subclass's own ``__init__(...)`` signature is left to the subclass.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self {
            state: TypifierState::Python(None),
        }
    }

    /// Reject a subclass that defines ``typify`` in its own body, and a
    /// subclass of a native typifier that defines ``assign`` or ``source_forcefield``.
    ///
    /// ``typify`` is the only writer of :meth:`forcefield`; an override would
    /// silently bypass the output. A native typifier (``OplsAaTypifier``,
    /// ``Mmff94Typifier``, …) types in Rust and never calls a Python ``assign``
    /// or ``source_forcefield``, so overriding either on its subclass would be silently
    /// ignored. ``typing.final`` is only a static check.
    #[classmethod]
    #[pyo3(signature = (**kwargs))]
    fn __init_subclass__(
        cls: &Bound<'_, PyType>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let py = cls.py();
        let own = cls.getattr(intern!(py, "__dict__"))?;
        if own.contains(intern!(py, "typify"))? {
            return Err(PyTypeError::new_err(format!(
                "{} defines typify; a Typifier subclass implements assign (and optionally \
                 source_forcefield) only — typify is the base's and the only writer of forcefield()",
                cls.name()?
            )));
        }
        let native = cls.is_subclass_of::<PyOplsAaTypifier>()?
            || cls.is_subclass_of::<PyMmff94Typifier>()?
            || cls.is_subclass_of::<PyMmff94sTypifier>()?
            || cls.is_subclass_of::<PyElementTypifier>()?
            || cls.is_subclass_of::<atd::PyAtdTypifier>()?
            || cls.is_subclass_of::<gaff::PyGaffTypifier>()?;
        if native {
            for hook in ["assign", "source_forcefield"] {
                if own.contains(hook)? {
                    return Err(PyTypeError::new_err(format!(
                        "{} defines {hook} on a native typifier, whose typify runs in Rust \
                         and never calls it; subclass molpy.ff.typifier.Typifier to supply \
                         your own {hook}",
                        cls.name()?
                    )));
                }
            }
        }
        PySuper::new(&py.get_type::<Self>(), cls.as_any())?.call_method(
            intern!(py, "__init_subclass__"),
            (),
            kwargs,
        )?;
        Ok(())
    }

    /// Type ``graph`` and return what it assigns, as a :class:`TypeAssignment`.
    ///
    /// The one hook a subclass implements. ``assign`` may write intermediate
    /// results (generated topology, perceived bond types) onto the graph it is
    /// given; :meth:`typify` always gives it a private copy. On a native
    /// typifier this runs the Rust matcher on ``graph``.
    ///
    /// Raises
    /// ------
    /// NotImplementedError
    ///     On the base, when a subclass does not implement ``assign``.
    /// ValueError
    ///     If a native matcher cannot match the graph.
    fn assign(&self, graph: &Bound<'_, PyAny>) -> PyResult<PyTypeAssignment> {
        match &self.state {
            TypifierState::Native(typing) => {
                let graph = graph.cast::<PyAtomistic>()?;
                let inner = typing
                    .typifier()
                    .assign(graph.borrow_mut().core_mut())
                    .map_err(typing_err)?;
                Ok(PyTypeAssignment { inner })
            }
            TypifierState::Python(_) => Err(PyNotImplementedError::new_err(
                "Typifier.assign must be implemented by a concrete typifier",
            )),
        }
    }

    /// Type ``mol``: do not override; the only writer of :meth:`forcefield`.
    ///
    /// Copies ``mol`` (as ``copy.copy(mol)``), calls :meth:`assign` on the copy and
    /// writes the returned :class:`TypeAssignment` onto the copy and the output force
    /// field — stamping every annotation and defining every type. ``mol`` is
    /// never touched.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///     The typed copy.
    ///
    /// Raises
    /// ------
    /// NotImplementedError
    ///     If the typifier has no ``assign``.
    /// TypeError
    ///     If ``assign`` returns something other than a :class:`TypeAssignment`.
    /// ValueError
    ///     If the match does not fit the graph or contradicts a definition the
    ///     output already holds. The output is then unchanged.
    fn typify(slf: &Bound<'_, Self>, mol: &Bound<'_, PyAtomistic>) -> PyResult<Py<PyAtomistic>> {
        let py = slf.py();
        let native = match &mut slf.borrow_mut().state {
            TypifierState::Native(typing) => {
                Some(typing.typify(mol.borrow().core()).map_err(typing_err)?)
            }
            TypifierState::Python(_) => None,
        };
        if let Some(typed) = native {
            return mol.borrow().derive(py, typed);
        }

        Self::seed(slf)?;
        let typed = {
            let mol = mol.try_borrow()?;
            mol.derive(py, mol.core().clone())?.into_bound(py)
        };
        let returned = slf.call_method1(intern!(py, "assign"), (&typed,))?;
        let matched = returned
            .cast::<PyTypeAssignment>()
            .map_err(|_| {
                PyTypeError::new_err(format!(
                    "assign must return a TypeAssignment, got {}",
                    returned.get_type()
                ))
            })?
            .get()
            .inner
            .clone();
        match &mut slf.borrow_mut().state {
            TypifierState::Python(Some(output)) => matched
                .apply_to(typed.borrow_mut().core_mut(), output)
                .map_err(PyValueError::new_err)?,
            TypifierState::Native(_) | TypifierState::Python(None) => {
                return Err(unseeded_output());
            }
        }
        Ok(typed.unbind())
    }

    /// The accumulated output — exactly the definitions :meth:`typify` has
    /// assigned — as an independent copy.
    ///
    /// Edits to the returned force field do not reach the typifier;
    /// :meth:`typify` is the only writer. Before the first ``typify`` this is
    /// the seeded empty output (see :meth:`source_forcefield`).
    fn forcefield(slf: &Bound<'_, Self>) -> PyResult<Py<PyForceField>> {
        Self::seed(slf)?;
        let output = match &slf.borrow().state {
            TypifierState::Native(typing) => typing.forcefield().clone(),
            TypifierState::Python(Some(output)) => output.clone(),
            TypifierState::Python(None) => return Err(unseeded_output()),
        };
        PyForceField::from_core(slf.py(), output)
    }

    /// The force field this typifier matches against, as an independent copy.
    ///
    /// The output starts as its empty likeness: the library's name and
    /// declared units and special_bonds, no styles or types. A Python subclass
    /// may override it; one that does not has no library (this raises
    /// ``NotImplementedError``), and its output starts as an empty force field
    /// named after the class, with nothing declared.
    fn source_forcefield(&self, py: Python<'_>) -> PyResult<Py<PyForceField>> {
        match &self.state {
            TypifierState::Native(typing) => {
                PyForceField::from_core(py, typing.typifier().source_forcefield().clone())
            }
            TypifierState::Python(_) => Err(PyNotImplementedError::new_err(
                "Typifier.source_forcefield is not implemented by this typifier",
            )),
        }
    }
}

/// What a typifier's ``assign`` assigns to one graph, exposed to Python as
/// ``molpy.ff.typifier.TypeAssignment`` (the Rust `TypeAssignment`).
///
/// Parameters
/// ----------
/// nodes : sequence of mapping
///     One mapping of ``key -> annotation`` per node, positional against
///     ``graph.atoms``. An empty mapping gives that node nothing.
/// links : mapping, optional
///     Relation kind to a sequence of mappings, positional against that
///     kind's own rows (an improper never shifts a dihedral position). A key
///     is a relation view class (``Bond``, ``Angle``, ``Dihedral``,
///     ``Improper``, ``Port``), rows as ``graph.bonds`` / ``graph.angles`` /
///     … / ``graph.ports``, or a block name (``"bonds"``, or a term block such
///     as ``"urey_bradleys"``), rows as ``graph.terms(block)``. A type
///     annotation under a kind defines a type of the category whose block
///     the kind is (``urey_bradleys`` → ``urey_bradley``). Any other key
///     raises ``TypeError``; a kind named twice raises ``ValueError``.
/// styles : sequence of (category, style, params), optional
///     Styles to declare, in order.
/// pairs : sequence of (style, name, endpoints, params), optional
///     Pair rows to define.
///
/// Notes
/// -----
/// An annotation is a ``str``, ``bool``, ``int`` or ``float`` (stamped, defines
/// nothing), or a type: ``(style, name, endpoints, params)``, which stamps
/// ``name`` and every param and defines the type ``name`` on ``endpoints``
/// (atom-type names; empty for an atom type) under the style. The name is
/// never read for endpoints. Param values are numbers or strings.
#[pyclass(
    module = "molpy.ff.typifier",
    name = "TypeAssignment",
    frozen,
    subclass
)]
pub struct PyTypeAssignment {
    inner: TypeAssignment,
}

impl PyTypeAssignment {
    /// One positional vector: a sequence of `key -> annotation` mappings.
    fn rows(rows: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<(String, Annotation)>>> {
        rows.try_iter()?
            .map(|row| {
                let row = row?;
                let row = row.cast::<PyMapping>()?;
                row.items()?
                    .iter()
                    .map(|item| {
                        let (key, value): (String, Bound<'_, PyAny>) = item.extract()?;
                        let annotation = Self::annotation(&value).map_err(|err| {
                            prefix_err(value.py(), err, &format!("annotation '{key}'"))
                        })?;
                        Ok((key, annotation))
                    })
                    .collect()
            })
            .collect()
    }

    /// A tuple is a type; anything else a stamped value (`bool` before `int`).
    fn annotation(value: &Bound<'_, PyAny>) -> PyResult<Annotation> {
        let Ok(tuple) = value.cast::<PyTuple>() else {
            return py_to_prop(value).map(Annotation::Value);
        };
        if tuple.len() != 4 {
            return Err(PyTypeError::new_err(format!(
                "a type annotation is (style, name, endpoints, params); got a {}-tuple",
                tuple.len()
            )));
        }
        let (style, name, endpoints, params): (String, String, Vec<String>, Bound<'_, PyDict>) =
            tuple.extract()?;
        Ok(Annotation::Type {
            style,
            name,
            endpoints,
            params: params_from_dict(Some(&params))?,
        })
    }
}

/// `err` with `context` prepended to its message, same exception type.
fn prefix_err(py: Python<'_>, err: PyErr, context: &str) -> PyErr {
    PyErr::from_type(err.get_type(py), format!("{context}: {}", err.value(py)))
}

#[pymethods]
impl PyTypeAssignment {
    #[new]
    #[pyo3(
        signature = (nodes, links = None, *, styles = Vec::new(), pairs = Vec::new()),
        text_signature = "(nodes, links=None, *, styles=(), pairs=())"
    )]
    fn new(
        nodes: &Bound<'_, PyAny>,
        links: Option<&Bound<'_, PyAny>>,
        styles: Vec<(String, String, Bound<'_, PyDict>)>,
        pairs: Vec<(String, String, Vec<String>, Bound<'_, PyDict>)>,
    ) -> PyResult<Self> {
        let mut inner = TypeAssignment {
            nodes: Self::rows(nodes)?,
            ..TypeAssignment::default()
        };
        if let Some(links) = links {
            for item in links.cast::<PyMapping>()?.items()?.iter() {
                let (key, rows): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item.extract()?;
                let kind = match RelationClass::atomistic_kind(&key) {
                    Some(kind) => kind.to_owned(),
                    None => key.extract::<String>().map_err(|_| {
                        PyTypeError::new_err(format!(
                            "links keys must be Atomistic relation view classes or relation \
                             kind names, got {key}"
                        ))
                    })?,
                };
                if inner.links.contains_key(&kind) {
                    return Err(PyValueError::new_err(format!(
                        "links name relation kind '{kind}' more than once"
                    )));
                }
                let rows = Self::rows(&rows).map_err(|err| prefix_err(rows.py(), err, &kind))?;
                inner.links.insert(kind, rows);
            }
        }
        for (category, name, params) in styles {
            let params = params_from_dict(Some(&params))?;
            inner.styles.push((category, name, params));
        }
        for (style, name, endpoints, params) in pairs {
            let params = params_from_dict(Some(&params))?;
            inner.pairs.push((style, name, endpoints, params));
        }
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        let m = &self.inner;
        let links: Vec<String> = m
            .links
            .iter()
            .map(|(kind, rows)| format!("{kind}={}", rows.len()))
            .collect();
        format!(
            "TypeAssignment(nodes={}, links={{{}}}, styles={}, pairs={})",
            m.nodes.len(),
            links.join(", "),
            m.styles.len(),
            m.pairs.len()
        )
    }
}

/// Bind one MMFF front door to Python.
///
/// MMFF94 and MMFF94s are the same engine over two parameter sets, and molrs
/// exposes them as two **named types** rather than one type with a variant flag —
/// so the binder mirrors that shape exactly: two `#[pyclass]`es, each wrapping its
/// own core typifier, generated from one forwarding body so they cannot drift.
macro_rules! py_mmff_front_door {
    (
        $(#[$doc:meta])*
        $py_ty:ident, $core:ty, $name:literal
    ) => {
        $(#[$doc])*
        #[pyclass(module = "molpy.ff.typifier", name = $name, extends = PyTypifier, subclass)]
        pub struct $py_ty;

        #[pymethods]
        impl $py_ty {
            /// Create the typifier with its embedded parameter tables.
            ///
            /// Never fails: the parameter set is compiled into the extension module.
            #[new]
            fn new() -> (Self, PyTypifier) {
                (Self, PyTypifier::native(<$core>::new()))
            }

            fn __repr__(slf: PyRef<'_, Self>) -> String {
                format!("{}(forcefield='{}')", $name, slf.as_super().source_forcefield_name())
            }
        }
    };
}

py_mmff_front_door! {
    /// MMFF94 atom-type assigner.
    ///
    /// Exposed to Python as `molpy.ff.typifier.Mmff94Typifier`.
    ///
    /// Loads the embedded MMFF94 parameter tables at construction time. Use
    /// :meth:`typify` to label a molecular graph (atom types, partial charges, and
    /// the per-instance force constants the kernels read: ``koop`` (md*A*rad^-2)
    /// on the improper rows, ``v1``/``v2``/``v3`` on the dihedral rows), then
    /// compile it through :meth:`forcefield` — the definitions typing assigned —
    /// the standard route, shared with every other force field in molrs.
    /// :meth:`typify` raises ``ValueError`` when atom types cannot be determined
    /// (e.g. unsupported elements).
    ///
    /// See :class:`Mmff94sTypifier` for the "static" variant used in energy
    /// minimization.
    ///
    /// # References
    ///
    /// - Halgren, T.A. (1996). J. Comput. Chem. 17, 490-519.
    ///
    /// Examples
    /// --------
    /// >>> typifier = Mmff94Typifier()
    /// >>> frame = typifier.typify(mol).to_frame()          # labels + charges
    /// >>> frame["pairs"] = molpy.ff.potential.intramolecular_pairs(frame)
    /// >>> pots = molpy.ff.compile.PotentialCompiler(typifier.forcefield()).compile(frame)
    PyMmff94Typifier, Mmff94Typifier, "Mmff94Typifier"
}

py_mmff_front_door! {
    /// MMFF94s ("static") atom-type assigner and potential builder.
    ///
    /// Exposed to Python as `molpy.ff.typifier.Mmff94sTypifier`.
    ///
    /// Identical to :class:`Mmff94Typifier` except on delocalised trivalent
    /// nitrogen (MMFF numeric types 10 ``NC=O`` and 40 ``NC=C``), where MMFF94s
    /// re-parameterises 11 out-of-plane rows and 42 torsion rows so the nitrogen
    /// minimizes to a **planar** geometry — the one seen in crystal structures.
    ///
    /// The mechanism is the out-of-plane force constant ``koop`` (md*A*rad^-2) that
    /// :meth:`typify` bakes onto the improper rows. The kernel evaluates
    /// ``E_oop = 0.5 * 143.9325 * koop * chi**2`` with ``chi`` the Wilson
    /// out-of-plane angle in radians, so ``koop > 0`` makes the planar centre an
    /// energy minimum. MMFF94s sets it to ``+0.015`` (type 10) / ``+0.030``
    /// (type 40); MMFF94's values on those rows run from ``-0.033`` to ``+0.004``.
    ///
    /// All 95 atom types, and every bond / angle / stretch-bend / vdW / charge
    /// parameter, are shared with MMFF94 — so a molecule with no such nitrogen gets
    /// bit-for-bit the same answer from both classes.
    ///
    /// # References
    ///
    /// - Halgren, T.A. (1999). J. Comput. Chem. 20, 720-729. (MMFF94s)
    ///
    /// Examples
    /// --------
    /// >>> typifier = Mmff94sTypifier()
    /// >>> typifier.forcefield().name
    /// 'MMFF94s'
    PyMmff94sTypifier, Mmff94sTypifier, "Mmff94sTypifier"
}

/// OPLS-AA atom-type assigner and potential builder.
///
/// Exposed to Python as `molpy.ff.typifier.OplsAaTypifier`. It loads the embedded canonical
/// OPLS-AA parameter set by default, or reads one XML source at construction.
/// :meth:`typify` returns a typed :class:`Atomistic` (``ValueError`` when atom
/// typing fails); :meth:`forcefield` holds the definitions it assigned.
///
/// Parameters
/// ----------
/// source : os.PathLike, bytes or molpy.io.Text, optional
///     An OPLS-AA XML file (a path), or its whole content (``bytes`` /
///     :class:`molpy.io.Text`); a plain ``str`` is refused as ambiguous.
///     ``None`` uses the embedded canonical OPLS-AA table.
/// strict : bool, default True
///     When True, a bonded term with no force-field match is an error. When
///     False, such terms are skipped (left unparametrized).
///
/// Examples
/// --------
/// >>> typifier = OplsAaTypifier()
/// >>> typed = typifier.typify(mol)        # typed Atomistic
/// >>> # compose: typify → to_frame → intramolecular_pairs → PotentialCompiler(forcefield()).compile
#[pyclass(module = "molpy.ff.typifier", name = "OplsAaTypifier", extends = PyTypifier, subclass)]
pub struct PyOplsAaTypifier;

#[pymethods]
impl PyOplsAaTypifier {
    /// Create an OPLS-AA typifier from embedded data, XML text, or an XML path.
    #[new]
    #[pyo3(signature = (source = None, *, strict = true))]
    fn new(source: Option<PySource<'_>>, strict: bool) -> PyResult<(Self, PyTypifier)> {
        let typifier = match source {
            Some(source) => molrs::io::openmm_xml::OpenmmXmlReader::new()
                .read_opls_typifier(source.source())
                .map_err(|e| PyValueError::new_err(format!("OPLS-AA XML source: {e}")))?,
            None => OplsAaTypifier::oplsaa(),
        }
        .with_strict(strict);
        Ok((Self, PyTypifier::native(typifier)))
    }

    fn __repr__(slf: PyRef<'_, Self>) -> String {
        format!(
            "OplsAaTypifier(forcefield='{}')",
            slf.as_super().source_forcefield_name()
        )
    }
}

/// Element typing: ``type`` labels from atomic numbers alone, with no force
/// field — ``molpy.ff.typifier.ElementTypifier``.
///
/// :meth:`typify` returns a typed :class:`Atomistic` whose atoms carry
/// ``type`` = the symbol their ``atomic_number`` names (e.g. ``"C"``) and
/// whose bonds, angles and dihedrals carry their endpoint symbols joined
/// with ``-`` in the byte-wise smaller
/// orientation (bond O–H is ``"H-O"``). It is for writers that need type
/// labels (LAMMPS data) on a molecule no force field has typed;
/// :meth:`forcefield` stays empty. Takes no arguments.
///
/// Raises
/// ------
/// ValueError
///     From :meth:`typify`, when an atom has no valid ``atomic_number`` or
///     is the wildcard (``0``), or the molecule has impropers.
///
/// Examples
/// --------
/// >>> typed = molpy.ff.typifier.ElementTypifier().typify(water)
/// >>> list(typed.to_frame()["bonds"]["type"])
/// ['H-O', 'H-O']
#[pyclass(module = "molpy.ff.typifier", name = "ElementTypifier", extends = PyTypifier, subclass)]
pub struct PyElementTypifier;

#[pymethods]
impl PyElementTypifier {
    #[new]
    fn new() -> (Self, PyTypifier) {
        (Self, PyTypifier::native(ElementTypifier::new()))
    }

    fn __repr__(&self) -> String {
        "ElementTypifier()".to_owned()
    }
}

/// Replace ``mol``'s ``cmaps`` terms, in place, by the CMAP crossterms its
/// dihedrals and ``forcefield``'s ``cmap`` rows define; return their number.
///
/// A crossterm is five atoms ``(a, b, c, d, e)`` whose dihedrals
/// ``(a, b, c, d)`` and ``(b, c, d, e)`` are both dihedrals of ``mol`` (in
/// either stored direction) and whose atom ``type`` labels equal a ``cmap``
/// row's ``itom`` … ``mtom`` in that order — never reversed. Each is one
/// ``cmaps`` term with ``type`` = the row's name; ``mol.to_frame()`` writes
/// them as the ``cmaps`` block.
///
/// Raises
/// ------
/// ValueError
///     Cmap rows in two styles, two rows on the same five types, a path that
///     matches one row forward and another backward, or an untyped dihedral
///     atom. ``mol`` is unchanged.
#[pyfunction]
#[pyo3(name = "assign_cmaps")]
pub fn assign_cmaps_py(mol: &Bound<'_, PyAtomistic>, forcefield: &PyForceField) -> PyResult<usize> {
    molrs::ff::typifier::cmap::assign_cmaps(mol.try_borrow_mut()?.core_mut(), &forcefield.inner)
        .map_err(typing_err)
}

/// Register `molpy.ff.typifier` (the base before its subclasses).
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTypifier>()?;
    m.add_class::<PyTypeAssignment>()?;
    m.add_class::<PyMmff94Typifier>()?;
    m.add_class::<PyMmff94sTypifier>()?;
    m.add_class::<PyOplsAaTypifier>()?;
    m.add_class::<atd::PyAtdTypifier>()?;
    m.add_class::<gaff::PyGaffTypifier>()?;
    m.add_class::<PyElementTypifier>()?;
    crate::add_function(
        m,
        "molpy.ff.typifier",
        wrap_pyfunction!(assign_cmaps_py, m)?,
    )?;
    Ok(())
}
