//! Placement of template copies: the [`Placer`] trait, the site-anchored
//! [`SitePlacer`] and the chain-growing [`GrowthPlacer`].

use std::fmt;

use crate::core::CenterError;
use crate::core::RelationId;
use crate::core::{Graph, GraphAccess};
use crate::op::Vec3;
use crate::op::vec3::sub;
use crate::op::{Rigid, alignment_axis_angle, axis_angle, compose_rigid, transform_point};

/// The bond from a site to its already-placed parent.
#[derive(Debug, Clone, Copy)]
pub struct ParentJoin {
    /// This copy's port on the bond to the parent (a port of the template).
    pub port: RelationId,
    /// The parent copy's placed anchor on that bond (Å).
    pub anchor: Vec3,
    /// The parent copy's placed leaving-group handle on that bond (Å).
    pub handle: Vec3,
    /// The covalent radius of the parent's anchor element (Å), when its
    /// element is known. With the copy's anchor element, it sets the new
    /// bond's length; without either, the copy's anchor takes the handle's
    /// place.
    pub anchor_radius: Option<f64>,
}

/// What a placer knows about one site.
#[derive(Debug, Clone)]
pub struct PlaceSite {
    /// The site's position (Å), when the site graph carries one.
    pub position: Option<Vec3>,
    /// The orienter's turn of the template, about its centroid
    /// (`Rigid::IDENTITY` without an orienter).
    pub turn: Rigid,
    /// The bond to the parent site, placed before this one; `None` for the
    /// first site placed in its molecule.
    pub parent: Option<ParentJoin>,
    /// Every other bond from this site to a site placed before it: a
    /// ladder's second bond to the same parent, or a ring, crosslink or
    /// periodic closure onto a site placed earlier (its anchor and handle
    /// taken in the image the bond names). A placer that honours them closes
    /// every bond at once; empty for a single join.
    pub extra_joins: Vec<ParentJoin>,
}

/// Turns one template and one site into the copy's pose.
///
/// A placer answers one question for the
/// [`Assembler`](crate::builder::Assembler): where does this copy go? It
/// returns the copy's whole pose, a rigid motion ([`Rigid`], `p' = R p + t`)
/// of the template, and may rotate as well as translate (operator,
/// 2026-09-28: "the placer puts a fragment at a point"). The assembler calls
/// it once per site, parents before children, so a placer can build on the
/// copies already placed.
///
/// The trait is generic over the template's graph type `G` (a
/// [`MolGraph`](crate::core::MolGraph), an
/// [`Atomistic`](crate::core::Atomistic) or a
/// [`CoarseGrain`](crate::core::CoarseGrain)), so an implementor reads the
/// template through that type's own vocabulary and an
/// [`Assembler<G>`](crate::builder::Assembler) holds it as a
/// `Box<dyn Placer<G>>`. [`SitePlacer`] and [`GrowthPlacer`] place every
/// graph type.
///
/// Implementors are `Send + Sync` so an
/// [`Assembler`](crate::builder::Assembler) holding one can cross threads.
pub trait Placer<G: Graph>: Send + Sync {
    /// The pose of a copy of `template` at `site`: the whole rigid motion
    /// taking template coordinates (Å) to world coordinates, `site.turn`
    /// included.
    ///
    /// # Errors
    ///
    /// A [`PlaceError`] naming why the copy cannot be placed.
    fn place(&self, template: &G, site: &PlaceSite) -> Result<Rigid, PlaceError>;
}

/// Puts each copy's centroid on its site.
///
/// With the template's centroid `R_c` (the mass-weighted mean of its
/// node positions, as [`Atomistic::center`](crate::core::Atomistic::center) computes it, Å), the
/// orienter's turn `T` (which fixes `R_c`) and the site position `p` (Å),
/// the pose is `x ↦ T x + (p − R_c)`: turned, then moved so its centre of
/// mass lands on `p`. This is the translation step of geometric backmapping
/// (Wassenaar et al., *J. Chem. Theory Comput.* **10**, 676–690 (2014),
/// doi:10.1021/ct400617g); overlap is left to a later relaxation.
///
/// # Examples
///
/// ```
/// use molrs::builder::{PlaceSite, Placer, SitePlacer};
/// use molrs::op::Rigid;
/// use molrs::core::keys;
/// use molrs::core::Atomistic;
/// use molrs::props;
/// use molrs::core::Element;
///
/// // Two carbons 1.5 Å apart: the centroid is (0.75, 0, 0).
/// let mut template = Atomistic::new();
/// for x in [0.0, 1.5] {
///     let c = template.add_atom(props! {
///         keys::ATOMIC_NUMBER => Element::C,
///         keys::X => x,
///         keys::Y => 0.0,
///         keys::Z => 0.0,
///     }).unwrap();
///     template.set_atom(c, keys::MASS, 12.0).unwrap();
/// }
///
/// let site = PlaceSite { position: Some([10.0, 0.0, 0.0]), turn: Rigid::IDENTITY, parent: None, extra_joins: Vec::new() };
/// let pose = SitePlacer::new().place(&template, &site).unwrap();
/// assert_eq!(pose.translation, [9.25, 0.0, 0.0]);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SitePlacer;

impl SitePlacer {
    /// The placer. It takes no configuration.
    pub fn new() -> Self {
        Self
    }
}

impl<G: Graph> Placer<G> for SitePlacer {
    /// `x ↦ T x + (p − R_c)`.
    ///
    /// # Errors
    ///
    /// [`PlaceError::NoPosition`] without a site position;
    /// [`PlaceError::NonFinitePoint`] for a NaN or infinite position;
    /// [`PlaceError::Template`] when the template has no centroid.
    fn place(&self, template: &G, site: &PlaceSite) -> Result<Rigid, PlaceError> {
        let template = template.graph();
        let p = site.position.ok_or(PlaceError::NoPosition)?;
        if !p.iter().all(|c| c.is_finite()) {
            return Err(PlaceError::NonFinitePoint);
        }
        let center = template
            .centroid(&template.node_ids().collect::<Vec<_>>())
            .map_err(PlaceError::Template)?;
        let shift = Rigid {
            rotation: Rigid::IDENTITY.rotation,
            translation: sub(p, center),
        };
        Ok(compose_rigid(&shift, &site.turn))
    }
}

/// Grows each molecule copy by copy from its first site.
///
/// The first copy of a molecule keeps its (turned) template pose, moved so
/// its centroid lands on the site position when the site graph has
/// one. Every later copy joins its parent through the port on their bond:
/// the copy is rotated, about its anchor `a`, by the smallest rotation taking
/// its anchor→handle direction `h − a` onto the parent's handle→anchor
/// direction `A − H`, then moved so `a` lies on the line from the parent's
/// anchor `A` through `H`, as far from `A` as the two anchors' covalent
/// radii add up to (both elements known), or on `H` itself (either one
/// unknown, as in a coarse-grained template). The new bond thus points back
/// along the parent's leaving bond at a covalent length; bond angles and
/// overlaps are left to a later minimisation.
///
/// A site with more than one bond to sites placed before it (a ladder, a
/// ring or crosslink closure, a bridge between two placed sites, a periodic
/// closure) is placed by one rigid least-squares fit closing every such bond
/// at once: the best geometry a rigid copy can give. A rigid copy cannot
/// always close them all — a bridge whose anchors are 8 Å apart cannot span
/// 10 Å — and a later minimisation relaxes what the fit leaves. When the
/// geometry must close, [`with_max_fit_rmsd`](Self::with_max_fit_rmsd) makes
/// a fit that misses its targets by more a refusal.
///
/// # Examples
///
/// ```
/// use molrs::builder::{GrowthPlacer, ParentJoin, PlaceSite, Placer};
/// use molrs::op::{Rigid, transform_point};
/// use molrs::core::keys;
/// use molrs::core::BondNumber;
/// use molrs::core::Atomistic;
/// use molrs::core::PortKind;
/// use molrs::props;
/// use molrs::core::Element;
///
/// // C with its `<` hydrogen at +x.
/// let mut unit = Atomistic::new();
/// let c = unit.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::C,
///     keys::X => 0.0,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// }).unwrap();
/// let h = unit.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::H,
///     keys::X => 1.0,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// }).unwrap();
/// for atom in [c, h] {
///     unit.set_atom(atom, keys::MASS, 1.0).unwrap();
/// }
/// unit.add_bond(c, h, props! {}).unwrap();
/// let port = unit.add_port(c, h, PortKind::Left, "", BondNumber::Single).unwrap();
///
/// // The parent's anchor sits at the origin, its handle at (0, 2, 0).
/// let parent = ParentJoin { port, anchor: [0.0; 3], handle: [0.0, 2.0, 0.0], anchor_radius: None };
/// let site = PlaceSite { position: None, turn: Rigid::IDENTITY, parent: Some(parent), extra_joins: Vec::new() };
/// let pose = GrowthPlacer::new().place(&unit, &site).unwrap();
///
/// // The copy's C lands on the parent's handle, its H points back down.
/// let (c_at, h_at) = (transform_point(&pose, [0.0; 3]), transform_point(&pose, [1.0, 0.0, 0.0]));
/// assert!((c_at[1] - 2.0).abs() < 1e-12 && (h_at[1] - 1.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct GrowthPlacer {
    max_fit_rmsd: Option<f64>,
}

impl GrowthPlacer {
    /// The placer; every fit is accepted.
    pub fn new() -> Self {
        Self::default()
    }

    /// Refuse a site closing several bonds at once when its rigid fit misses
    /// the bond targets by more than `max` Å, root mean square.
    ///
    /// # Errors
    ///
    /// [`PlaceError::Tolerance`] unless `max` is finite and not negative.
    pub fn with_max_fit_rmsd(self, max: f64) -> Result<Self, PlaceError> {
        if !(max.is_finite() && max >= 0.0) {
            return Err(PlaceError::Tolerance(max));
        }
        Ok(Self {
            max_fit_rmsd: Some(max),
        })
    }

    /// The misfit a several-bond fit may have (Å, rms); `None` accepts any.
    pub fn max_fit_rmsd(&self) -> Option<f64> {
        self.max_fit_rmsd
    }
}

impl<G: Graph> Placer<G> for GrowthPlacer {
    /// The joined pose for a site with a parent; the turned template, on the
    /// site position when there is one, for the first site of a molecule.
    ///
    /// # Errors
    ///
    /// [`PlaceError::NonFinitePoint`] for a NaN or infinite position;
    /// [`PlaceError::Template`] when a first site with a position has a
    /// template without a centroid; [`PlaceError::Port`] when the
    /// parent bond's port does not read back, has an atom without
    /// coordinates, or its anchor and handle coincide.
    fn place(&self, template: &G, site: &PlaceSite) -> Result<Rigid, PlaceError> {
        let Some(join) = site.parent else {
            return match site.position {
                Some(p) => SitePlacer.place(
                    template,
                    &PlaceSite {
                        position: Some(p),
                        ..site.clone()
                    },
                ),
                None => Ok(site.turn),
            };
        };
        if !site.extra_joins.is_empty() {
            return join_all(template, site, join, self.max_fit_rmsd);
        }
        let graph = template.graph();
        let port = graph
            .get_port(join.port)
            .map_err(|e| PlaceError::Port(e.to_string()))?;
        let position = |atom| -> Result<Vec3, PlaceError> {
            graph
                .get_node(atom)
                .map_err(|e| PlaceError::Port(e.to_string()))?
                .position()
                .ok_or_else(|| PlaceError::Port("a port atom has no x/y/z".to_owned()))
        };
        let a = transform_point(&site.turn, position(port.anchor)?);
        let h = transform_point(&site.turn, position(port.handle)?);
        let own_radius = graph
            .get_node(port.anchor)
            .ok()
            .and_then(|atom| atom.atomic_number())
            .and_then(crate::core::Element::by_number)
            .map(|element| f64::from(element.covalent_radius()));
        let (from, to) = (sub(h, a), sub(join.anchor, join.handle));
        let rotation = match alignment_axis_angle(from, to) {
            Some((axis, angle)) => axis_angle(axis, angle).expect("alignment gives a unit axis"),
            None if crate::op::vec3::normalize(from).is_some()
                && crate::op::vec3::normalize(to).is_some() =>
            {
                Rigid::IDENTITY.rotation
            }
            None => {
                return Err(PlaceError::Port(
                    "an anchor and its handle coincide".to_owned(),
                ));
            }
        };
        let turn_about_anchor = crate::op::rotation_about(rotation, a);
        // The new bond runs along the parent's anchor → handle line, as long
        // as the two anchors' covalent radii; with an element unknown, the
        // copy's anchor takes the handle's place.
        let target = bond_target(&join, own_radius)?;
        let shift = Rigid {
            rotation: Rigid::IDENTITY.rotation,
            translation: sub(target, a),
        };
        Ok(compose_rigid(
            &shift,
            &compose_rigid(&turn_about_anchor, &site.turn),
        ))
    }
}

/// Where a copy's anchor goes on a bond to its parent: along the parent's
/// anchor → handle line, as far as the two anchors' covalent radii add up to,
/// or on the handle when either element is unknown.
fn bond_target(join: &ParentJoin, own_radius: Option<f64>) -> Result<Vec3, PlaceError> {
    match (join.anchor_radius, own_radius) {
        (Some(parent_r), Some(own_r)) => {
            let along = crate::op::vec3::normalize(sub(join.handle, join.anchor))
                .ok_or_else(|| PlaceError::Port("an anchor and its handle coincide".to_owned()))?;
            let length = parent_r + own_r;
            Ok([
                join.anchor[0] + along[0] * length,
                join.anchor[1] + along[1] * length,
                join.anchor[2] + along[2] * length,
            ])
        }
        _ => Ok(join.handle),
    }
}

/// The covalent radius of a port's anchor element, when it has one.
fn anchor_radius<G: Graph>(template: &G, anchor: crate::core::NodeId) -> Option<f64> {
    template
        .graph()
        .get_node(anchor)
        .ok()
        .and_then(|atom| atom.atomic_number())
        .and_then(crate::core::Element::by_number)
        .map(|element| f64::from(element.covalent_radius()))
}

/// The pose that closes every join at once: a least-squares rigid fit
/// (``superpose``) laying each copy anchor on its bond target and each copy
/// handle on the line back to the joined anchor. A fit the points leave
/// under-determined falls back to the first join alone. With a `max`, the
/// pose's root-mean-square distance from the targets must be at most `max` Å.
fn join_all<G: Graph>(
    template: &G,
    site: &PlaceSite,
    first: ParentJoin,
    max: Option<f64>,
) -> Result<Rigid, PlaceError> {
    let graph = template.graph();
    let mut reference = Vec::new();
    let mut untouched = Vec::new();
    let mut target = Vec::new();
    let mut weights = Vec::new();
    for join in std::iter::once(first).chain(site.extra_joins.iter().copied()) {
        let port = graph
            .get_port(join.port)
            .map_err(|e| PlaceError::Port(e.to_string()))?;
        let position = |atom| -> Result<Vec3, PlaceError> {
            graph
                .get_node(atom)
                .map_err(|e| PlaceError::Port(e.to_string()))?
                .position()
                .ok_or_else(|| PlaceError::Port("a port atom has no x/y/z".to_owned()))
        };
        let (raw_a, raw_h) = (position(port.anchor)?, position(port.handle)?);
        untouched.extend([raw_a, raw_h]);
        let a = transform_point(&site.turn, raw_a);
        let h = transform_point(&site.turn, raw_h);
        // The anchor goes on its bond target; the handle points back from
        // there toward the parent anchor, at the copy's own anchor–handle
        // length, so a consistent geometry is fitted exactly.
        let goal = bond_target(&join, anchor_radius(template, port.anchor))?;
        let back = crate::op::vec3::normalize(sub(join.anchor, goal)).ok_or_else(|| {
            PlaceError::Port("a bond target coincides with the parent anchor".to_owned())
        })?;
        let reach = crate::op::vec3::norm(sub(h, a));
        reference.extend([a, h]);
        target.extend([
            goal,
            [
                goal[0] + back[0] * reach,
                goal[1] + back[1] * reach,
                goal[2] + back[2] * reach,
            ],
        ]);
        weights.extend([1.0, 1.0]);
    }
    let fit = crate::op::superpose(&reference, &target, &weights, crate::op::DEFAULT_GAP_TOL)
        .map_err(|e| PlaceError::Port(format!("the joins to the parent cannot be fitted: {e}")))?;
    let pose = if matches!(fit.freedom, crate::op::Freedom::Unique) {
        compose_rigid(&fit.rigid, &site.turn)
    } else {
        let single = PlaceSite {
            extra_joins: Vec::new(),
            ..site.clone()
        };
        Placer::<G>::place(&GrowthPlacer::new(), template, &single)?
    };
    let Some(max) = max else {
        return Ok(pose);
    };
    // The misfit of the pose itself, from the template's own coordinates.
    let squared: f64 = untouched
        .iter()
        .zip(&target)
        .map(|(&r, &t)| {
            let d = sub(transform_point(&pose, r), t);
            d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        })
        .sum();
    let rmsd = (squared / untouched.len() as f64).sqrt();
    if rmsd.is_nan() || rmsd > max {
        return Err(PlaceError::Misfit { rmsd, max });
    }
    Ok(pose)
}

/// Why a [`Placer`] could not place a template.
#[derive(Debug, Clone, PartialEq)]
pub enum PlaceError {
    /// The template has no centroid.
    Template(CenterError),
    /// The site has no position and the placer needs one.
    NoPosition,
    /// The site position has a NaN or infinite coordinate.
    NonFinitePoint,
    /// The port joining the parent does not read back or has no direction.
    Port(String),
    /// The rigid fit closing a site's bonds misses their targets by `rmsd`
    /// Å, more than the `max` allowed.
    Misfit {
        /// The fit's root-mean-square misfit (Å).
        rmsd: f64,
        /// The misfit allowed (Å).
        max: f64,
    },
    /// A misfit tolerance that is negative or not finite.
    Tolerance(f64),
}

impl fmt::Display for PlaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Template(e) => write!(f, "the template has no centroid: {e}"),
            Self::NoPosition => write!(f, "the site has no position"),
            Self::NonFinitePoint => write!(f, "the site position is not finite"),
            Self::Port(why) => write!(f, "the joining port is unusable: {why}"),
            Self::Misfit { rmsd, max } => write!(
                f,
                "the rigid fit closing the site's bonds misses them by {rmsd:.3} Å rms, more than {max} Å"
            ),
            Self::Tolerance(max) => {
                write!(f, "max_fit_rmsd must be finite and not negative, got {max}")
            }
        }
    }
}

impl std::error::Error for PlaceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Template(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GrowthPlacer, ParentJoin, PlaceError, PlaceSite, Placer, SitePlacer};
    use crate::core::Atomistic;
    use crate::core::BondNumber;
    use crate::core::Element;
    use crate::core::PortKind;
    use crate::core::RelationId;
    use crate::core::keys;
    use crate::op::{Rigid, rotation_about, transform_point};
    use crate::props;

    const TOL: f64 = 1e-12;

    /// Monomer M: C0 (0,0,0), C1 (1.5,0,0), H0 (−1,0,0), H1 (2.5,0,0); masses
    /// 12, 12, 1, 1 (or none). Centre of mass x = 19.5 / 26 = 0.75. Ports
    /// (C0, H0, `<`) and (C1, H1, `>`).
    fn monomer() -> (Atomistic, RelationId, RelationId) {
        let mut m = Atomistic::new();
        let mut ids = Vec::new();
        for (element, x) in [
            (Element::C, 0.0),
            (Element::C, 1.5),
            (Element::H, -1.0),
            (Element::H, 2.5),
        ] {
            let id = m
                .add_atom(props! {
                    keys::ATOMIC_NUMBER => element,
                    keys::X => x,
                    keys::Y => 0.0,
                    keys::Z => 0.0,
                })
                .unwrap();
            ids.push(id);
        }
        for (a, b) in [(0, 1), (0, 2), (1, 3)] {
            m.add_bond(ids[a], ids[b], props! {}).expect("bond");
        }
        let l = m
            .add_port(ids[0], ids[2], PortKind::Left, "", BondNumber::Single)
            .expect("port");
        let r = m
            .add_port(ids[1], ids[3], PortKind::Right, "", BondNumber::Single)
            .expect("port");
        (m, l, r)
    }

    fn site(position: Option<[f64; 3]>, parent: Option<ParentJoin>) -> PlaceSite {
        PlaceSite {
            extra_joins: Vec::new(),
            position,
            turn: Rigid::IDENTITY,
            parent,
        }
    }

    fn close(got: [f64; 3], want: [f64; 3]) {
        for d in 0..3 {
            assert!((got[d] - want[d]).abs() < TOL, "want {want:?}, got {got:?}");
        }
    }

    #[test]
    fn site_placer_moves_the_centre_of_mass_onto_the_site() {
        let (m, _, _) = monomer();
        let pose = SitePlacer::new()
            .place(&m, &site(Some([10.0, 0.0, 0.0]), None))
            .expect("placeable");
        assert_eq!(pose.translation, [9.25, 0.0, 0.0]);
        assert_eq!(pose.rotation, Rigid::IDENTITY.rotation);
    }

    #[test]
    fn site_placer_keeps_the_orienter_turn() {
        let (m, _, _) = monomer();
        let rz = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let turned = PlaceSite {
            position: Some([0.0; 3]),
            turn: rotation_about(rz, [0.75, 0.0, 0.0]),
            parent: None,
            extra_joins: Vec::new(),
        };
        let pose = SitePlacer::new().place(&m, &turned).expect("placeable");
        // C1 (1.5,0,0) is 0.75 along +x of the centre: after the quarter turn
        // it sits 0.75 along +y of the site.
        close(transform_point(&pose, [1.5, 0.0, 0.0]), [0.0, 0.75, 0.0]);
    }

    #[test]
    fn site_placer_refuses_a_site_without_a_finite_position() {
        let (m, _, _) = monomer();
        assert_eq!(
            SitePlacer::new().place(&m, &site(None, None)),
            Err(PlaceError::NoPosition)
        );
        assert_eq!(
            SitePlacer::new().place(&m, &site(Some([f64::NAN, 0.0, 0.0]), None)),
            Err(PlaceError::NonFinitePoint)
        );
    }

    #[test]
    fn growth_placer_keeps_the_first_copy_where_the_template_is() {
        let (m, _, _) = monomer();
        let pose = GrowthPlacer::new()
            .place(&m, &site(None, None))
            .expect("first copy");
        assert_eq!(pose, Rigid::IDENTITY);
    }

    #[test]
    fn growth_placer_puts_the_anchor_on_the_parent_handle_pointing_back() {
        let (m, left, _) = monomer();
        // Parent's `>` anchor at (0,0,5), its handle at (0,0,6): the child's
        // `<` anchor C0 lands on (0,0,6) and its handle H0 on the line back,
        // at (0,0,5).
        let join = ParentJoin {
            port: left,
            anchor: [0.0, 0.0, 5.0],
            handle: [0.0, 0.0, 6.0],
            anchor_radius: None,
        };
        let pose = GrowthPlacer::new()
            .place(&m, &site(None, Some(join)))
            .expect("joined");
        close(transform_point(&pose, [0.0; 3]), [0.0, 0.0, 6.0]);
        close(transform_point(&pose, [-1.0, 0.0, 0.0]), [0.0, 0.0, 5.0]);
    }

    #[test]
    fn growth_placer_sets_the_new_bond_to_the_covalent_length() {
        let (m, left, _) = monomer();
        // A carbon parent anchor at (0,0,5) whose C–H handle sits 1.09 Å up:
        // the child's carbon anchor lands 0.76 + 0.76 Å from it, on that line,
        // and its own handle points back toward the parent.
        let join = ParentJoin {
            port: left,
            anchor: [0.0, 0.0, 5.0],
            handle: [0.0, 0.0, 6.09],
            anchor_radius: Some(f64::from(Element::C.covalent_radius())),
        };
        let pose = GrowthPlacer::new()
            .place(&m, &site(None, Some(join)))
            .expect("joined");
        let length = 2.0 * f64::from(Element::C.covalent_radius());
        close(transform_point(&pose, [0.0; 3]), [0.0, 0.0, 5.0 + length]);
        close(
            transform_point(&pose, [-1.0, 0.0, 0.0]),
            [0.0, 0.0, 4.0 + length],
        );
    }

    /// A ladder end: two carbons, each with a `<` hydrogen along −x; the
    /// second carbon sits at +z, so a copy joined through the first port
    /// alone would leave the second bond open.
    fn ladder_end() -> (Atomistic, RelationId, RelationId) {
        let mut m = Atomistic::new();
        let mut ids = Vec::new();
        for (element, x, z) in [
            (Element::C, 0.0, 0.0),
            (Element::H, -1.09, 0.0),
            (Element::C, 0.0, 1.4),
            (Element::H, -1.09, 1.4),
        ] {
            ids.push(
                m.add_atom(props! {
                    keys::ATOMIC_NUMBER => element,
                    keys::X => x,
                    keys::Y => 0.0,
                    keys::Z => z,
                })
                .unwrap(),
            );
        }
        m.add_bond(ids[0], ids[1], props! {}).unwrap();
        m.add_bond(ids[2], ids[3], props! {}).unwrap();
        m.add_bond(ids[0], ids[2], props! {}).unwrap();
        let a = m
            .add_port(ids[0], ids[1], PortKind::Left, "a", BondNumber::Single)
            .unwrap();
        let b = m
            .add_port(ids[2], ids[3], PortKind::Left, "b", BondNumber::Single)
            .unwrap();
        (m, a, b)
    }

    #[test]
    fn growth_placer_closes_every_bond_to_the_same_parent() {
        let (m, a, b) = ladder_end();
        let r = f64::from(Element::C.covalent_radius());
        // The parent's two anchors sit 1.4 Å apart along +y, their handles
        // along +x: both new bonds must come out one C–C length long.
        let first = ParentJoin {
            port: a,
            anchor: [0.0; 3],
            handle: [1.09, 0.0, 0.0],
            anchor_radius: Some(r),
        };
        let second = ParentJoin {
            port: b,
            anchor: [0.0, 1.4, 0.0],
            handle: [1.09, 1.4, 0.0],
            anchor_radius: Some(r),
        };
        let mut joined = site(None, Some(first));
        joined.extra_joins = vec![second];
        // A consistent ladder closes exactly: even a vanishing tolerance
        // accepts it.
        let placer = GrowthPlacer::new().with_max_fit_rmsd(1e-9).unwrap();
        let pose = placer.place(&m, &joined).expect("joined");
        let length = 2.0 * r;
        let c0 = transform_point(&pose, [0.0, 0.0, 0.0]);
        let c1 = transform_point(&pose, [0.0, 0.0, 1.4]);
        let d0 = crate::op::vec3::norm(crate::op::vec3::sub(c0, first.anchor));
        let d1 = crate::op::vec3::norm(crate::op::vec3::sub(c1, second.anchor));
        assert!((d0 - length).abs() < 1e-6, "first bond {d0}, want {length}");
        assert!(
            (d1 - length).abs() < 1e-6,
            "second bond {d1}, want {length}"
        );
    }

    /// The same ladder end, its parent anchors 3 Å apart instead of 1.4: no
    /// rigid pose closes both bonds, and the misfit decides.
    #[test]
    fn growth_placer_refuses_a_fit_beyond_its_tolerance() {
        let (m, a, b) = ladder_end();
        let r = f64::from(Element::C.covalent_radius());
        let join = |port, y: f64| ParentJoin {
            port,
            anchor: [0.0, y, 0.0],
            handle: [1.09, y, 0.0],
            anchor_radius: Some(r),
        };
        let mut joined = site(None, Some(join(a, 0.0)));
        joined.extra_joins = vec![join(b, 3.0)];
        let tight = GrowthPlacer::new().with_max_fit_rmsd(0.1).unwrap();
        match Placer::<Atomistic>::place(&tight, &m, &joined).unwrap_err() {
            PlaceError::Misfit { rmsd, max } => {
                assert_eq!(max, 0.1);
                assert!(rmsd > 0.1, "{rmsd}");
            }
            other => panic!("expected a misfit, got {other}"),
        }
        let loose = GrowthPlacer::new().with_max_fit_rmsd(5.0).unwrap();
        assert!(Placer::<Atomistic>::place(&loose, &m, &joined).is_ok());
        // Without a tolerance the best fit is taken, whatever it misses by.
        assert!(Placer::<Atomistic>::place(&GrowthPlacer::new(), &m, &joined).is_ok());
    }

    #[test]
    fn a_tolerance_must_be_finite_and_not_negative() {
        for bad in [-0.1, f64::NAN, f64::INFINITY] {
            assert!(GrowthPlacer::new().with_max_fit_rmsd(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn growth_placer_refuses_a_join_whose_parent_anchor_and_handle_coincide() {
        let (m, left, _) = monomer();
        let join = ParentJoin {
            port: left,
            anchor: [1.0; 3],
            handle: [1.0; 3],
            anchor_radius: None,
        };
        let err = GrowthPlacer::new()
            .place(&m, &site(None, Some(join)))
            .expect_err("no direction");
        assert!(matches!(err, PlaceError::Port(_)), "{err:?}");
    }
}
