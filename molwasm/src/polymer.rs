//! Polymer ensembles — the WASM face of `molrs::builder`'s
//! generative-BigSMILES model.
//!
//! | JS | molrs |
//! |----|-------|
//! | `Distribution.schulzZimm(mw, mn)` … | `Distribution::SchulzZimm { .. }` … |
//! | `new ChainSize(distribution, measure)` | `ChainSize::new` |
//! | `new Fragment(name, template, weights?)` | `Fragment::new` |
//! | `new Terminal(kind, label?, order?, weights?)` | `Terminal::new` |
//! | `StochasticObjectBuilder` … `.build()` | `StochasticObject::new` |
//! | `MoleculeBuilder` … `.build()` | `Molecule::new` |
//! | `SystemBuilder` … `.build()` | `System::new` |
//! | `molecule.sample(seed)`, `system.sample(seed)` | `Molecule::sample`, `System::sample` |
//!
//! A template and a site graph cross as a [`Frame`]: a fragment's template is
//! the frame of a ported molecule (`SmilesIr.fromFragment(body).toTemplate()`),
//! and a sample is the frame of its site graph — one row of `atoms` per
//! placed fragment, its `bead_type` the fragment's name. Objects that hold a
//! list are assembled with a builder (`add…()` then `build()`), because a JS
//! array of class instances cannot be handed to Rust without consuming them;
//! every refusal is thrown by `build()`, with molrs's message.
//!
//! # Typical workflow (JavaScript)
//!
//! ```js
//! const unit = (name, body) =>
//!   new Fragment(name, SmilesIr.fromFragment(body).toTemplate());
//! const chain = new StochasticObjectBuilder();
//! chain.addRepeatUnit(unit("E", "[<]CC[>]"));
//! chain.addEndGroup(unit("Me>", "C[>]"));
//! chain.addEndGroup(unit("<Me", "[<]C"));
//! chain.setSize(new ChainSize(Distribution.poisson(20), "repeat_units"));
//! const molecule = new MoleculeBuilder();
//! molecule.addObject(chain.build());
//! const sites = molecule.build().sample(42n); // Frame: one site per fragment
//! ```

use std::str::FromStr;

use molrs::builder::{
    Amount as RsAmount, ChainSize as RsChainSize, Distribution as RsDistribution,
    Fragment as RsFragment, Measure, Molecule as RsMolecule, Part, PolymerError, SiteWeight,
    StochasticObject as RsStochasticObject, System as RsSystem, Terminal as RsTerminal,
};
use molrs::core::{Atomistic, BondNumber, PortKind};
use wasm_bindgen::prelude::*;

use crate::core::Frame;

fn js_error(error: PolymerError) -> JsValue {
    JsValue::from_str(&error.to_string())
}

fn measure(name: &str) -> Result<Measure, JsValue> {
    match name {
        "repeat_units" => Ok(Measure::RepeatUnits),
        "heavy_atom_mass" => Ok(Measure::HeavyAtomMass),
        "mass" => Ok(Measure::Mass),
        other => Err(JsValue::from_str(&format!(
            "measure must be 'repeat_units', 'heavy_atom_mass' or 'mass', not '{other}'"
        ))),
    }
}

fn measure_name(measure: Measure) -> String {
    match measure {
        Measure::RepeatUnits => "repeat_units",
        Measure::HeavyAtomMass => "heavy_atom_mass",
        Measure::Mass => "mass",
    }
    .to_owned()
}

/// `templates` as a JS `Map` of name → [`Frame`].
fn library(
    templates: std::collections::HashMap<String, Atomistic>,
) -> Result<js_sys::Map, JsValue> {
    let map = js_sys::Map::new();
    let mut names: Vec<&String> = templates.keys().collect();
    names.sort();
    for name in names {
        let frame = templates[name]
            .to_frame()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        map.set(&JsValue::from_str(name), &Frame::from_rs(frame)?.into());
    }
    Ok(map)
}

/// A chain-size distribution of generative BigSMILES — one static
/// constructor per distribution, parameters as the notation writes them.
/// What a drawn value counts is the [`ChainSize`] measure it sits in.
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct Distribution {
    inner: RsDistribution,
}

#[wasm_bindgen]
impl Distribution {
    /// `P(k) = a² k (1 − a)^(k − 1)`, `0 < a < 1`.
    #[wasm_bindgen(js_name = florySchulz)]
    pub fn flory_schulz(a: f64) -> Distribution {
        Distribution {
            inner: RsDistribution::FlorySchulz { a },
        }
    }

    /// Gamma of mean `mn` and weight average `mw`, `mw > mn > 0`.
    #[wasm_bindgen(js_name = schulzZimm)]
    pub fn schulz_zimm(mw: f64, mn: f64) -> Distribution {
        Distribution {
            inner: RsDistribution::SchulzZimm { mw, mn },
        }
    }

    /// Normal, `sigma ≥ 0`.
    pub fn gauss(mu: f64, sigma: f64) -> Distribution {
        Distribution {
            inner: RsDistribution::Gauss { mu, sigma },
        }
    }

    /// Uniform over `[low, high]`, `0 ≤ low ≤ high`.
    pub fn uniform(low: f64, high: f64) -> Distribution {
        Distribution {
            inner: RsDistribution::Uniform { low, high },
        }
    }

    /// Log-normal of number average `mn` and dispersity `d > 1`.
    #[wasm_bindgen(js_name = logNormal)]
    pub fn log_normal(mn: f64, d: f64) -> Distribution {
        Distribution {
            inner: RsDistribution::LogNormal { mn, d },
        }
    }

    /// Poisson of mean `n > 0`.
    pub fn poisson(n: f64) -> Distribution {
        Distribution {
            inner: RsDistribution::Poisson { n },
        }
    }
}

/// Size targets drawn from a [`Distribution`], counted in `measure`:
/// `"repeat_units"`, `"heavy_atom_mass"` or `"mass"` (masses in g/mol).
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct ChainSize {
    inner: RsChainSize,
}

#[wasm_bindgen]
impl ChainSize {
    /// # Errors
    ///
    /// Throws for parameters outside the distribution's range, or an
    /// unknown measure.
    #[wasm_bindgen(constructor)]
    pub fn new(distribution: &Distribution, measure: &str) -> Result<ChainSize, JsValue> {
        let inner =
            RsChainSize::new(distribution.inner, self::measure(measure)?).map_err(js_error)?;
        Ok(ChainSize { inner })
    }

    /// What a target counts.
    #[wasm_bindgen(getter)]
    pub fn measure(&self) -> String {
        measure_name(self.inner.measure())
    }
}

/// A port weight as JS writes it: a number (its reactivity) or an array of
/// numbers (its pairing weight with each port of the stochastic object).
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum WeightArg {
    Single(f64),
    List(Vec<f64>),
}

/// One ported template under the name sites refer to it by, with a
/// generation weight per port.
#[wasm_bindgen]
pub struct Fragment {
    inner: RsFragment,
}

#[wasm_bindgen]
impl Fragment {
    /// A fragment named `name` over the ported molecule in `template` (the
    /// frame `SmilesIr.fromFragment(body).toTemplate()` gives). `weights` is
    /// one entry per port, a number or an array of numbers; omitted, every
    /// port weighs 1.
    ///
    /// # Errors
    ///
    /// Throws when the frame is not a ported molecule, or molrs refuses the
    /// fragment (weight count, a negative weight, …).
    #[wasm_bindgen(constructor)]
    pub fn new(name: &str, template: &Frame, weights: JsValue) -> Result<Fragment, JsValue> {
        let template = template.with_frame(|rs| {
            Atomistic::from_frame(rs)
                .map_err(|e| JsValue::from_str(&format!("Frame → Atomistic: {e}")))
        })?;
        let weights = if weights.is_undefined() || weights.is_null() {
            vec![SiteWeight::default(); template.n_ports()]
        } else {
            let args: Vec<WeightArg> = serde_wasm_bindgen::from_value(weights)
                .map_err(|e| JsValue::from_str(&format!("weights: {e}")))?;
            args.into_iter()
                .map(|w| match w {
                    WeightArg::Single(x) => SiteWeight::Single(x),
                    WeightArg::List(xs) => SiteWeight::List(xs),
                })
                .collect()
        };
        let inner = RsFragment::new(name, template, weights).map_err(js_error)?;
        Ok(Fragment { inner })
    }

    /// The name sites refer to this fragment by.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.inner.name().to_owned()
    }

    /// The number of ports.
    #[wasm_bindgen(getter, js_name = nPorts)]
    pub fn n_ports(&self) -> usize {
        self.inner.ports().len()
    }

    /// Summed non-hydrogen atomic mass, g/mol.
    #[wasm_bindgen(getter, js_name = heavyAtomMass)]
    pub fn heavy_atom_mass(&self) -> f64 {
        self.inner.heavy_atom_mass()
    }

    /// Mass with every port bonded (heavy atoms plus kept hydrogens), g/mol.
    #[wasm_bindgen(getter)]
    pub fn mass(&self) -> f64 {
        self.inner.mass()
    }
}

/// The descriptor at one end of a stochastic object, standing for the port
/// of the part on that side.
#[wasm_bindgen]
pub struct Terminal {
    inner: RsTerminal,
}

#[wasm_bindgen]
impl Terminal {
    /// A terminal of `kind` (`"$"`, `"<"` or `">"`), `label` (default `""`),
    /// bond `order` (default 1) and optional list `weights`, one per port of
    /// the object.
    ///
    /// # Errors
    ///
    /// Throws for an unknown kind, `"!"`, an order outside 1…4, or weights
    /// that are empty, negative or not finite.
    #[wasm_bindgen(constructor)]
    pub fn new(
        kind: &str,
        label: Option<String>,
        order: Option<u32>,
        weights: Option<Vec<f64>>,
    ) -> Result<Terminal, JsValue> {
        let kind = PortKind::from_str(kind).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let inner = RsTerminal::new(
            kind,
            label.unwrap_or_default(),
            BondNumber::from_code(order.unwrap_or(1)),
            weights,
        )
        .map_err(js_error)?;
        Ok(Terminal { inner })
    }
}

/// Repeat units and end groups grown to a size drawn from a [`ChainSize`].
#[wasm_bindgen]
pub struct StochasticObject {
    inner: RsStochasticObject,
}

/// Collects what a [`StochasticObject`] is made of; `build()` constructs it.
#[wasm_bindgen]
#[derive(Default)]
pub struct StochasticObjectBuilder {
    repeat_units: Vec<RsFragment>,
    end_groups: Vec<RsFragment>,
    left: Option<RsTerminal>,
    right: Option<RsTerminal>,
    size: Option<RsChainSize>,
}

#[wasm_bindgen]
impl StochasticObjectBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> StochasticObjectBuilder {
        StochasticObjectBuilder::default()
    }

    /// Add a repeat unit; list weights index ports in the order added.
    #[wasm_bindgen(js_name = addRepeatUnit)]
    pub fn add_repeat_unit(&mut self, fragment: &Fragment) {
        self.repeat_units.push(fragment.inner.clone());
    }

    /// Add an end group (a fragment with exactly one port).
    #[wasm_bindgen(js_name = addEndGroup)]
    pub fn add_end_group(&mut self, fragment: &Fragment) {
        self.end_groups.push(fragment.inner.clone());
    }

    /// The terminal standing for the port of the part before the object.
    #[wasm_bindgen(js_name = setLeft)]
    pub fn set_left(&mut self, terminal: &Terminal) {
        self.left = Some(terminal.inner.clone());
    }

    /// The terminal standing for the port of the part after the object.
    #[wasm_bindgen(js_name = setRight)]
    pub fn set_right(&mut self, terminal: &Terminal) {
        self.right = Some(terminal.inner.clone());
    }

    /// The size distribution; without one the object cannot be sampled.
    #[wasm_bindgen(js_name = setSize)]
    pub fn set_size(&mut self, size: &ChainSize) {
        self.size = Some(size.inner);
    }

    /// Construct the object.
    ///
    /// # Errors
    ///
    /// Throws what `StochasticObject::new` refuses.
    pub fn build(&self) -> Result<StochasticObject, JsValue> {
        RsStochasticObject::new(
            self.repeat_units.clone(),
            self.end_groups.clone(),
            self.left.clone(),
            self.right.clone(),
            self.size,
        )
        .map(|inner| StochasticObject { inner })
        .map_err(js_error)
    }
}

/// Fixed fragments and stochastic objects joined end to end.
#[wasm_bindgen]
pub struct Molecule {
    pub(crate) inner: RsMolecule,
}

#[wasm_bindgen]
impl Molecule {
    /// Every fragment the molecule may place, as a `Map` of name →
    /// template [`Frame`].
    ///
    /// # Errors
    ///
    /// Throws if a template cannot be written as a frame.
    pub fn library(&self) -> Result<js_sys::Map, JsValue> {
        library(self.inner.library())
    }

    /// One sample as a site-graph [`Frame`]: an `atoms` row per placed
    /// fragment (`bead_type` its name) and a `bonds` row per join with
    /// `src_port` / `dst_port`. Deterministic in `seed`.
    ///
    /// # Errors
    ///
    /// Throws for an object without a size, or growth that cannot place or
    /// cap a fragment.
    pub fn sample(&self, seed: u64) -> Result<Frame, JsValue> {
        let sites = self.inner.sample(seed).map_err(js_error)?;
        Frame::from_rs(
            sites
                .to_frame()
                .map_err(|e| JsValue::from_str(&e.to_string()))?,
        )
    }
}

/// Collects the parts of a [`Molecule`], in order; `build()` constructs it.
#[wasm_bindgen]
#[derive(Default)]
pub struct MoleculeBuilder {
    parts: Vec<Part>,
}

#[wasm_bindgen]
impl MoleculeBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> MoleculeBuilder {
        MoleculeBuilder::default()
    }

    /// Append a fragment placed once (a prefix, a connector, a suffix).
    #[wasm_bindgen(js_name = addFragment)]
    pub fn add_fragment(&mut self, fragment: &Fragment) {
        self.parts.push(Part::Fixed(fragment.inner.clone()));
    }

    /// Append a stochastic object.
    #[wasm_bindgen(js_name = addObject)]
    pub fn add_object(&mut self, object: &StochasticObject) {
        self.parts.push(Part::Stochastic(object.inner.clone()));
    }

    /// Construct the molecule.
    ///
    /// # Errors
    ///
    /// Throws what `Molecule::new` refuses.
    pub fn build(&self) -> Result<Molecule, JsValue> {
        RsMolecule::new(self.parts.clone())
            .map(|inner| Molecule { inner })
            .map_err(js_error)
    }
}

/// One sampled molecule, as it reaches JS.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct Record {
    component: usize,
    n_fragments: usize,
    n_repeat_units: usize,
    heavy_atom_mass: f64,
    mass: f64,
}

/// One sample of a [`System`].
#[wasm_bindgen]
pub struct SystemSample {
    sites: molrs::core::CoarseGrain,
    molecules: Vec<Record>,
}

#[wasm_bindgen]
impl SystemSample {
    /// Every molecule's sites in one site-graph [`Frame`].
    ///
    /// # Errors
    ///
    /// Throws if the site graph cannot be written as a frame.
    pub fn sites(&self) -> Result<Frame, JsValue> {
        Frame::from_rs(
            self.sites
                .to_frame()
                .map_err(|e| JsValue::from_str(&e.to_string()))?,
        )
    }

    /// One record per molecule, in site order: `{ component, nFragments,
    /// nRepeatUnits, heavyAtomMass, mass }`.
    ///
    /// # Errors
    ///
    /// Throws if the records cannot be converted.
    pub fn molecules(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.molecules).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// Molecule kinds sampled until each reaches its amount of the total.
#[wasm_bindgen]
pub struct System {
    pub(crate) inner: RsSystem,
}

#[wasm_bindgen]
impl System {
    /// What the amounts count.
    #[wasm_bindgen(getter)]
    pub fn measure(&self) -> String {
        measure_name(self.inner.measure())
    }

    /// Each component's target, in the system's measure.
    pub fn targets(&self) -> Vec<f64> {
        self.inner.targets()
    }

    /// Every fragment any molecule may place, as a `Map` of name →
    /// template [`Frame`].
    ///
    /// # Errors
    ///
    /// Throws if a template cannot be written as a frame.
    pub fn library(&self) -> Result<js_sys::Map, JsValue> {
        library(self.inner.library())
    }

    /// Molecules of each component until it reaches its target; amounts are
    /// sampled as written. Deterministic in `seed`.
    ///
    /// # Errors
    ///
    /// Throws what sampling a molecule throws.
    pub fn sample(&self, seed: u64) -> Result<SystemSample, JsValue> {
        let sample = self.inner.sample(seed).map_err(js_error)?;
        Ok(SystemSample {
            sites: sample.sites,
            molecules: sample
                .molecules
                .iter()
                .map(|m| Record {
                    component: m.component,
                    n_fragments: m.n_fragments,
                    n_repeat_units: m.n_repeat_units,
                    heavy_atom_mass: m.heavy_atom_mass,
                    mass: m.mass,
                })
                .collect(),
        })
    }
}

/// Collects the components of a [`System`]; `build()` constructs it.
#[wasm_bindgen]
pub struct SystemBuilder {
    components: Vec<(RsMolecule, RsAmount)>,
    measure: Measure,
}

#[wasm_bindgen]
impl SystemBuilder {
    /// A builder whose amounts count `measure` (`"repeat_units"`,
    /// `"heavy_atom_mass"` or `"mass"`).
    ///
    /// # Errors
    ///
    /// Throws for an unknown measure.
    #[wasm_bindgen(constructor)]
    pub fn new(measure: &str) -> Result<SystemBuilder, JsValue> {
        Ok(SystemBuilder {
            components: Vec::new(),
            measure: self::measure(measure)?,
        })
    }

    /// Add a molecule kind in an absolute amount.
    #[wasm_bindgen(js_name = addTotal)]
    pub fn add_total(&mut self, molecule: &Molecule, value: f64) {
        self.components
            .push((molecule.inner.clone(), RsAmount::Total(value)));
    }

    /// Add a molecule kind as a percentage of the whole system.
    #[wasm_bindgen(js_name = addPercent)]
    pub fn add_percent(&mut self, molecule: &Molecule, percent: f64) {
        self.components
            .push((molecule.inner.clone(), RsAmount::Percent(percent)));
    }

    /// Construct the system.
    ///
    /// # Errors
    ///
    /// Throws what `System::new` refuses.
    pub fn build(&self) -> Result<System, JsValue> {
        RsSystem::new(self.components.clone(), self.measure)
            .map(|inner| System { inner })
            .map_err(js_error)
    }
}

#[cfg(all(test, feature = "smiles"))]
mod tests {
    use super::*;
    use crate::io::smiles::SmilesIr;
    use wasm_bindgen_test::*;

    fn fragment(name: &str, body: &str) -> Fragment {
        let template = SmilesIr::from_fragment(body)
            .unwrap()
            .to_template()
            .unwrap();
        Fragment::new(name, &template, JsValue::UNDEFINED).unwrap()
    }

    fn chain(n: f64) -> Molecule {
        let mut object = StochasticObjectBuilder::new();
        object.add_repeat_unit(&fragment("E", "[<]CC[>]"));
        object.add_end_group(&fragment("Me>", "C[>]"));
        object.add_end_group(&fragment("<Me", "[<]C"));
        object.set_size(&ChainSize::new(&Distribution::gauss(n, 0.0), "repeat_units").unwrap());
        let mut molecule = MoleculeBuilder::new();
        molecule.add_object(&object.build().unwrap());
        molecule.build().unwrap()
    }

    #[wasm_bindgen_test]
    fn a_fragment_reports_its_ports_and_masses() {
        let e = fragment("E", "[<]CC[>]");
        assert_eq!(e.name(), "E");
        assert_eq!(e.n_ports(), 2);
        assert!(e.mass() > e.heavy_atom_mass());
    }

    #[wasm_bindgen_test]
    fn a_molecule_samples_a_site_frame() {
        let molecule = chain(4.0);
        let sites = molecule.sample(1).unwrap();
        assert_eq!(sites.get("atoms").unwrap().n_rows().unwrap(), 6);
        assert_eq!(sites.get("bonds").unwrap().n_rows().unwrap(), 5);
        assert_eq!(molecule.library().unwrap().size(), 3);
    }

    #[wasm_bindgen_test]
    fn a_system_samples_to_its_total() {
        let mut builder = SystemBuilder::new("repeat_units").unwrap();
        builder.add_total(&chain(4.0), 20.0);
        let system = builder.build().unwrap();
        assert_eq!(system.targets(), vec![20.0]);
        let sample = system.sample(3).unwrap();
        assert_eq!(
            sample
                .sites()
                .unwrap()
                .get("atoms")
                .unwrap()
                .n_rows()
                .unwrap(),
            30
        );
        let records = js_sys::Array::from(&sample.molecules().unwrap());
        assert_eq!(records.length(), 5);
    }

    #[wasm_bindgen_test]
    fn refusals_are_thrown_with_the_core_message() {
        assert!(ChainSize::new(&Distribution::poisson(-1.0), "repeat_units").is_err());
        assert!(ChainSize::new(&Distribution::poisson(3.0), "monomers").is_err());
        assert!(Terminal::new("!", None, None, None).is_err());
        assert!(Terminal::new(">", None, Some(7), None).is_err());
        // Nothing before the object and no end group to start from.
        let mut object = StochasticObjectBuilder::new();
        object.add_repeat_unit(&fragment("E", "[$]CC[$]"));
        let mut molecule = MoleculeBuilder::new();
        molecule.add_object(&object.build().unwrap());
        assert!(molecule.build().is_err());
    }
}
