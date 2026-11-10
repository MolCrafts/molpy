//! A stochastic object and the growth that realises one copy of it.
//!
//! The growth is the generative-BigSMILES procedure (Schneider & de Pablo,
//! Digital Discovery 2024), stated over fragments and ports rather than atoms:
//!
//! 1. Draw a size target. With a part before the object, its port (the
//!    left [`Terminal`]) joins a repeat-unit port chosen by weight; with
//!    none, growth starts from an end group chosen by weight.
//! 2. While ports are open and the measured size is below the target: pick
//!    an open port by reactivity; a single-weighted port joins a repeat-unit
//!    port chosen by reactivity, a list-weighted port any port of the object
//!    chosen by its list; add that fragment. Every copy holds at least one
//!    repeat unit, whatever the target.
//! 3. With a right [`Terminal`], keep one open port it accepts open, for
//!    the part after the object — chosen by the terminal's list weights, or
//!    by reactivity. Growth never spends the last such port unless the
//!    fragment it adds brings another.
//! 4. Close every other open port with an end group chosen by weight.
//!
//! Two ports join only when [`Port::accepts`] holds: complementary kinds,
//! equal labels and equal orders.

use std::sync::Arc;

use rand::{Rng, RngExt};

use super::PolymerError;
use super::fragment::{Fragment, SiteWeight};
use super::size::{ChainSize, Measure};
use crate::core::{BondNumber, NodeId, Port, PortKind};
use crate::op::F;

/// The descriptor written at one end of a stochastic object, standing for
/// the port of the part on that side: `{[<]…[>]}` — `<` is the port of the
/// part before the object, `>` the port of the part after it.
#[derive(Debug, Clone, PartialEq)]
pub struct Terminal {
    kind: PortKind,
    label: String,
    order: BondNumber,
    weights: Option<Vec<F>>,
}

impl Terminal {
    /// A terminal of role `kind` (`$`, `<` or `>`), `label` (`""` when
    /// unnamed) and bond `order`.
    ///
    /// `weights` are generative-BigSMILES list weights (`[<|w₁ w₂ …|]`), one
    /// per port of the object in position order (repeat units, then end
    /// groups): on a left terminal the weight of entering the object at each
    /// port, on a right terminal the weight of leaving it from each. `None`
    /// uses the ports' own reactivities.
    ///
    /// # Errors
    ///
    /// [`PolymerError::InvalidObject`] for the squash kind `!`, an order
    /// that is not single … quadruple, or weights that are empty, negative
    /// or not finite.
    pub fn new(
        kind: PortKind,
        label: impl Into<String>,
        order: BondNumber,
        weights: Option<Vec<F>>,
    ) -> Result<Self, PolymerError> {
        let invalid =
            |reason: &str| Err(PolymerError::InvalidObject(format!("terminal: {reason}")));
        if kind == PortKind::Shared {
            return invalid("'!' joins nothing; a terminal is '$', '<' or '>'");
        }
        if order == BondNumber::Unknown {
            return invalid("the bond order is single, double, triple or quadruple");
        }
        if let Some(ws) = &weights
            && (ws.is_empty() || ws.iter().any(|w| !(w.is_finite() && *w >= 0.0)))
        {
            return invalid("weights are one or more finite, non-negative numbers");
        }
        Ok(Self {
            kind,
            label: label.into(),
            order,
            weights,
        })
    }

    /// The role of the outside port.
    pub fn kind(&self) -> PortKind {
        self.kind
    }

    /// Its label, `""` when unnamed.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The order of the bond it forms.
    pub fn order(&self) -> BondNumber {
        self.order
    }

    /// The list weights, if any.
    pub fn weights(&self) -> Option<&[F]> {
        self.weights.as_deref()
    }

    /// The outside port this terminal stands for, as a [`Port`] without
    /// atoms — only its kind, label and order take part in [`Port::accepts`].
    pub(super) fn port(&self) -> Port {
        Port {
            anchor: NodeId::default(),
            handle: NodeId::default(),
            kind: self.kind,
            label: self.label.clone(),
            order: self.order,
        }
    }

    /// Whether `port` describes the same outside port: kind, label and order.
    pub(super) fn describes(&self, port: &Port) -> bool {
        self.kind == port.kind && self.label == port.label && self.order == port.order
    }
}

/// Repeat units and end groups drawn into a chain of a random size.
///
/// Cloning is cheap: the object's data is shared.
#[derive(Debug, Clone)]
pub struct StochasticObject {
    data: Arc<ObjectData>,
}

#[derive(Debug)]
struct ObjectData {
    repeat_units: Vec<Fragment>,
    end_groups: Vec<Fragment>,
    left: Option<Terminal>,
    right: Option<Terminal>,
    size: Option<ChainSize>,
    plan: Plan,
}

/// What growth may do at each *position* — one port of the object, in the
/// order list weights index: the ports of the repeat units, then of the end
/// groups, each fragment's in port order. Everything here depends on the
/// object alone, so it is computed once, in [`StochasticObject::new`].
#[derive(Debug, Clone, Default)]
struct Plan {
    /// Each position's fragment (an index over repeat units then end
    /// groups) and port.
    owner: Vec<(usize, usize)>,
    /// Each fragment's first position.
    first: Vec<usize>,
    /// Each position's reactivity.
    reactivity: Vec<F>,
    /// The positions an open port at each position may grow onto, with
    /// positive weights: by its list over every accepting position, or by
    /// reactivity over the accepting repeat-unit positions.
    grow: Vec<Vec<(usize, F)>>,
    /// The end-group positions that cap an open port at each position, by
    /// reactivity.
    caps: Vec<Vec<(usize, F)>>,
    /// Where a part before the object enters: the repeat-unit positions the
    /// left terminal accepts, by its list or their reactivity.
    entry: Vec<(usize, F)>,
    /// Where growth starts with nothing before the object: the end-group
    /// positions, by reactivity.
    start: Vec<(usize, F)>,
    /// The weight of leaving the object from each position, for the
    /// positions the right terminal accepts.
    exit: Vec<Option<F>>,
    /// Whether each position's fragment has *another* port the right
    /// terminal accepts — placing it keeps a way out.
    keeps_exit: Vec<bool>,
}

/// One copy of the object: the fragments placed and how they join.
#[derive(Debug, Default)]
pub(super) struct Growth {
    /// The fragments, in the order they were added.
    pub instances: Vec<Fragment>,
    /// Each joined pair `((instance, port), (instance, port))`.
    pub joins: Vec<((usize, usize), (usize, usize))>,
    /// The port that joins the part before the object, if one does.
    pub entry: Option<(usize, usize)>,
    /// The port kept open for the part after the object, if the object has
    /// a right terminal.
    pub exit: Option<(usize, usize)>,
    /// How many of `instances` are repeat units.
    pub n_repeat_units: usize,
}

impl StochasticObject {
    /// An object drawing from `repeat_units` and `end_groups`, entered and
    /// left through the `left` and `right` terminals, grown to a target drawn
    /// from `size`.
    ///
    /// An object without a `size` describes the chemistry (its templates)
    /// but cannot be sampled — BigSMILES, as opposed to generative BigSMILES.
    ///
    /// # Errors
    ///
    /// [`PolymerError::InvalidObject`] for no repeat unit; an end group with
    /// a port count other than one; a list weight — on a port or a terminal
    /// — whose length is not the object's port count; a left terminal no
    /// repeat-unit port can be entered through, or a right terminal no
    /// repeat-unit port accepts; a repeat unit that adds nothing to the
    /// measure `size` counts.
    pub fn new(
        repeat_units: Vec<Fragment>,
        end_groups: Vec<Fragment>,
        left: Option<Terminal>,
        right: Option<Terminal>,
        size: Option<ChainSize>,
    ) -> Result<Self, PolymerError> {
        let invalid = |reason: String| Err(PolymerError::InvalidObject(reason));
        if repeat_units.is_empty() {
            return invalid("a stochastic object needs at least one repeat unit".to_owned());
        }
        if let Some(group) = end_groups.iter().find(|g| g.ports().len() != 1) {
            return invalid(format!(
                "end group '{}' has {} ports; an end group has exactly one",
                group.name(),
                group.ports().len()
            ));
        }
        let mut object = Self {
            data: Arc::new(ObjectData {
                repeat_units,
                end_groups,
                left,
                right,
                size,
                plan: Plan::default(),
            }),
        };
        let n: usize = object.fragments().map(|f| f.ports().len()).sum();
        let lists = object
            .fragments()
            .flat_map(|fragment| fragment.weights().iter().map(move |w| (fragment.name(), w)))
            .filter_map(|(name, w)| match w {
                SiteWeight::List(ws) => Some((format!("a port of '{name}'"), ws.len())),
                SiteWeight::Single(_) => None,
            })
            .chain(
                [
                    ("the left terminal", &object.data.left),
                    ("the right terminal", &object.data.right),
                ]
                .into_iter()
                .filter_map(|(what, t)| Some((what.to_owned(), t.as_ref()?.weights()?.len()))),
            );
        for (what, len) in lists {
            if len != n {
                return invalid(format!(
                    "{what} lists {len} weights; the object has {n} ports"
                ));
            }
        }
        let plan = object.planned();
        Arc::get_mut(&mut object.data)
            .expect("the object was built above and is not shared yet")
            .plan = plan;
        if let Some(left) = &object.data.left
            && !object.data.plan.entry.iter().any(|&(_, w)| w > 0.0)
        {
            return invalid(format!(
                "no repeat-unit port can be entered through the left terminal {}{}",
                left.kind.as_str(),
                left.label
            ));
        }
        if let Some(right) = &object.data.right {
            let n_repeat: usize = object
                .data
                .repeat_units
                .iter()
                .map(|f| f.ports().len())
                .sum();
            if !object.data.plan.exit[..n_repeat]
                .iter()
                .any(Option::is_some)
            {
                return invalid(format!(
                    "no repeat-unit port accepts the right terminal {}{}",
                    right.kind.as_str(),
                    right.label
                ));
            }
        }
        if let Some(size) = &object.data.size
            && let Some(inert) = object
                .repeat_units()
                .iter()
                .find(|f| f.size_in(size.measure(), true) <= 0.0)
        {
            return invalid(format!(
                "repeat unit '{}' adds nothing to the measure the size counts",
                inert.name()
            ));
        }
        Ok(object)
    }

    /// The repeat units.
    pub fn repeat_units(&self) -> &[Fragment] {
        &self.data.repeat_units
    }

    /// The end groups.
    pub fn end_groups(&self) -> &[Fragment] {
        &self.data.end_groups
    }

    /// The terminal standing for the port of the part before the object.
    pub fn left(&self) -> Option<&Terminal> {
        self.data.left.as_ref()
    }

    /// The terminal standing for the port of the part after the object.
    pub fn right(&self) -> Option<&Terminal> {
        self.data.right.as_ref()
    }

    /// The size distribution, `None` for an object that cannot be sampled.
    pub fn size(&self) -> Option<&ChainSize> {
        self.data.size.as_ref()
    }

    /// Every fragment, repeat units first.
    pub(super) fn fragments(&self) -> impl Iterator<Item = &Fragment> {
        self.data.repeat_units.iter().chain(&self.data.end_groups)
    }

    /// The least one copy can measure: its smallest repeat unit.
    pub(super) fn least(&self, measure: Measure) -> F {
        self.data
            .repeat_units
            .iter()
            .map(|f| f.size_in(measure, true))
            .fold(F::INFINITY, F::min)
    }

    fn fragment(&self, index: usize) -> &Fragment {
        match index.checked_sub(self.data.repeat_units.len()) {
            None => &self.data.repeat_units[index],
            Some(end) => &self.data.end_groups[end],
        }
    }

    fn is_end_group(&self, index: usize) -> bool {
        index >= self.data.repeat_units.len()
    }

    /// Compute the [`Plan`].
    fn planned(&self) -> Plan {
        let mut owner = Vec::new();
        let mut first = Vec::new();
        for (index, fragment) in self.fragments().enumerate() {
            first.push(owner.len());
            owner.extend((0..fragment.ports().len()).map(|port| (index, port)));
        }
        let port = |pos: usize| &self.fragment(owner[pos].0).ports()[owner[pos].1];
        let weight = |pos: usize| &self.fragment(owner[pos].0).weights()[owner[pos].1];
        let end_group = |pos: usize| self.is_end_group(owner[pos].0);
        let all = 0..owner.len();
        let reactivity: Vec<F> = all.clone().map(|pos| weight(pos).reactivity()).collect();

        let grow = all
            .clone()
            .map(|pos| {
                all.clone()
                    .filter(|&q| port(pos).accepts(port(q)))
                    .filter_map(|q| match weight(pos) {
                        SiteWeight::List(ws) => Some((q, ws[q])),
                        SiteWeight::Single(_) if !end_group(q) => Some((q, reactivity[q])),
                        SiteWeight::Single(_) => None,
                    })
                    .filter(|&(_, w)| w > 0.0)
                    .collect()
            })
            .collect();
        let caps = all
            .clone()
            .map(|pos| {
                all.clone()
                    .filter(|&q| end_group(q) && port(pos).accepts(port(q)))
                    .map(|q| (q, reactivity[q]))
                    .collect()
            })
            .collect();
        let entry = self.data.left.as_ref().map_or_else(Vec::new, |left| {
            let outside = left.port();
            all.clone()
                .filter(|&q| !end_group(q) && outside.accepts(port(q)))
                .map(|q| (q, left.weights().map_or(reactivity[q], |ws| ws[q])))
                .collect()
        });
        let start = all
            .clone()
            .filter(|&q| end_group(q))
            .map(|q| (q, reactivity[q]))
            .collect();
        let exit: Vec<Option<F>> = all
            .clone()
            .map(|pos| {
                let right = self.data.right.as_ref()?;
                right
                    .port()
                    .accepts(port(pos))
                    .then(|| right.weights().map_or(reactivity[pos], |ws| ws[pos]))
            })
            .collect();
        let keeps_exit = all
            .map(|pos| {
                let (index, own) = owner[pos];
                (0..self.fragment(index).ports().len())
                    .any(|other| other != own && exit[first[index] + other].is_some())
            })
            .collect();
        Plan {
            owner,
            first,
            reactivity,
            grow,
            caps,
            entry,
            start,
            exit,
            keeps_exit,
        }
    }

    /// Grow one copy. `entered` says a part before the object joins it
    /// through the left terminal.
    ///
    /// Each step costs the number of positions, not the number of open
    /// ports: open ports are kept per position, and what a port may join
    /// depends on its position alone.
    pub(super) fn grow<R: Rng + ?Sized>(
        &self,
        entered: bool,
        rng: &mut R,
    ) -> Result<Growth, PolymerError> {
        let size = self.data.size.as_ref().ok_or(PolymerError::Unsized)?;
        let target = size.sample(rng);
        let plan = &self.data.plan;
        let mut state = State::new(self, size.measure());

        // 1. The first fragment.
        if entered {
            let q = choose(&plan.entry, rng).expect("new checked the left terminal has an entry");
            state.growth.entry = Some(state.add(q));
        } else {
            let q = choose_or_uniform(&plan.start, rng).ok_or_else(|| {
                PolymerError::Stuck(
                    "nothing before the object and no end group to start it from".to_owned(),
                )
            })?;
            let (inst, _) = state.add(q);
            // The starting end group's own port is the growing end.
            state.open(inst, q);
        }

        // 2. Grow to the target — and to one repeat unit at the least.
        while state.growth.n_repeat_units == 0 || state.measured < target {
            // While no repeat unit is placed, an end group is not a partner.
            let need_unit = state.growth.n_repeat_units == 0;
            let allowed = |state: &State, pos: usize, q: usize| {
                !(need_unit && self.is_end_group(plan.owner[q].0))
                    && (!state.is_last_exit(pos) || plan.keeps_exit[q])
            };
            let options: Vec<(usize, F)> = (0..plan.owner.len())
                .filter(|&pos| !state.open[pos].is_empty())
                .filter(|&pos| plan.grow[pos].iter().any(|&(q, _)| allowed(&state, pos, q)))
                .map(|pos| (pos, plan.reactivity[pos] * state.open[pos].len() as F))
                .collect();
            let Some(pos) = choose(&options, rng) else {
                break;
            };
            let partners: Vec<(usize, F)> = plan.grow[pos]
                .iter()
                .copied()
                .filter(|&(q, _)| allowed(&state, pos, q))
                .collect();
            let q = choose(&partners, rng).expect("grow keeps positive weights only");
            let site = state.take(pos, rng);
            let joined = state.add(q);
            state.growth.joins.push((site, joined));
        }
        if state.growth.n_repeat_units == 0 {
            return Err(PolymerError::Stuck(
                "no repeat unit can be joined to the first fragment".to_owned(),
            ));
        }

        // 3. Keep one port open for the part after the object.
        if let Some(right) = &self.data.right {
            let options: Vec<(usize, F)> = (0..plan.owner.len())
                .filter(|&pos| !state.open[pos].is_empty())
                .filter_map(|pos| Some((pos, plan.exit[pos]? * state.open[pos].len() as F)))
                .collect();
            let pos = choose_or_uniform(&options, rng).ok_or_else(|| {
                PolymerError::Stuck(format!(
                    "no open port accepts the right terminal {}{}",
                    right.kind.as_str(),
                    right.label
                ))
            })?;
            state.growth.exit = Some(state.take(pos, rng));
        }

        // 4. Cap every other open port. An end group has one port, so capping
        //    opens nothing new.
        for (pos, sites) in std::mem::take(&mut state.open).into_iter().enumerate() {
            for inst in sites {
                let q = choose_or_uniform(&plan.caps[pos], rng).ok_or_else(|| {
                    let (index, port) = plan.owner[pos];
                    let port = &self.fragment(index).ports()[port];
                    PolymerError::Stuck(format!(
                        "open port {}{} of '{}' matches no end group",
                        port.kind.as_str(),
                        port.label,
                        self.fragment(index).name()
                    ))
                })?;
                let joined = state.add(q);
                state.growth.joins.push(((inst, plan.owner[pos].1), joined));
            }
        }
        Ok(state.growth)
    }
}

/// The growth in progress: what is placed, which ports are open, how large
/// it measures.
struct State<'a> {
    object: &'a StochasticObject,
    measure: Measure,
    growth: Growth,
    /// The instances with an open port at each position.
    open: Vec<Vec<usize>>,
    /// How many open ports the right terminal accepts.
    n_exits: usize,
    measured: F,
}

impl<'a> State<'a> {
    fn new(object: &'a StochasticObject, measure: Measure) -> Self {
        Self {
            object,
            measure,
            growth: Growth::default(),
            open: vec![Vec::new(); object.data.plan.owner.len()],
            n_exits: 0,
            measured: 0.0,
        }
    }

    /// Place the fragment of position `q`, joined through that position's
    /// port, every other port open; returns the joined `(instance, port)`.
    fn add(&mut self, q: usize) -> (usize, usize) {
        let plan = &self.object.data.plan;
        let (index, joined) = plan.owner[q];
        let fragment = self.object.fragment(index);
        let end_group = self.object.is_end_group(index);
        let inst = self.growth.instances.len();
        self.growth.instances.push(fragment.clone());
        self.growth.n_repeat_units += usize::from(!end_group);
        self.measured += fragment.size_in(self.measure, !end_group);
        for port in (0..fragment.ports().len()).filter(|&p| p != joined) {
            self.open(inst, plan.first[index] + port);
        }
        (inst, joined)
    }

    /// Mark `inst`'s port at position `pos` open.
    fn open(&mut self, inst: usize, pos: usize) {
        self.open[pos].push(inst);
        self.n_exits += usize::from(self.object.data.plan.exit[pos].is_some());
    }

    /// Close one open port at position `pos`, any of them with equal
    /// probability; returns it as `(instance, port)`.
    fn take<R: Rng + ?Sized>(&mut self, pos: usize, rng: &mut R) -> (usize, usize) {
        let pick = rng.random_range(0..self.open[pos].len());
        let inst = self.open[pos].swap_remove(pick);
        self.n_exits -= usize::from(self.object.data.plan.exit[pos].is_some());
        (inst, self.object.data.plan.owner[pos].1)
    }

    /// Whether the open port at `pos` is the only way left out of the
    /// object: joining it must bring another.
    fn is_last_exit(&self, pos: usize) -> bool {
        self.n_exits == 1 && self.object.data.plan.exit[pos].is_some()
    }
}

/// One of `options` drawn in proportion to its weight; `None` when no
/// weight is positive.
fn choose<T: Copy, R: Rng + ?Sized>(options: &[(T, F)], rng: &mut R) -> Option<T> {
    let total: F = options.iter().map(|(_, w)| w.max(0.0)).sum();
    if total.is_nan() || total <= 0.0 {
        return None;
    }
    let mut x = rng.random::<F>() * total;
    for &(item, w) in options {
        let w = w.max(0.0);
        if x < w {
            return Some(item);
        }
        x -= w;
    }
    options
        .iter()
        .rev()
        .find(|(_, w)| *w > 0.0)
        .map(|(item, _)| *item)
}

/// As [`choose`], but uniform when every weight is zero; `None` only for no
/// option at all. Used where a choice must be made — every open port gets
/// capped, whatever its weight.
fn choose_or_uniform<T: Copy, R: Rng + ?Sized>(options: &[(T, F)], rng: &mut R) -> Option<T> {
    if options.is_empty() {
        return None;
    }
    choose(options, rng).or_else(|| Some(options[rng.random_range(0..options.len())].0))
}
