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
}

/// What a placer knows about one site.
#[derive(Debug, Clone, Copy)]
pub struct PlaceSite {
    /// The site's position (Å), when the site graph carries one.
    pub position: Option<Vec3>,
    /// The orienter's turn of the template, about its centre of mass
    /// (`Rigid::IDENTITY` without an orienter).
    pub turn: Rigid,
    /// The bond to the parent site, placed before this one; `None` for the
    /// first site placed in its molecule.
    pub parent: Option<ParentJoin>,
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

/// Puts each copy's centre of mass on its site.
///
/// With the template's centre of mass `R_c` (the mass-weighted mean of its
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
/// // Two carbons 1.5 Å apart: the centre of mass is (0.75, 0, 0).
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
/// let site = PlaceSite { position: Some([10.0, 0.0, 0.0]), turn: Rigid::IDENTITY, parent: None };
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
    /// [`PlaceError::Template`] when the template has no centre of mass.
    fn place(&self, template: &G, site: &PlaceSite) -> Result<Rigid, PlaceError> {
        let template = template.graph();
        let p = site.position.ok_or(PlaceError::NoPosition)?;
        if !p.iter().all(|c| c.is_finite()) {
            return Err(PlaceError::NonFinitePoint);
        }
        let center = template
            .center(&template.node_ids().collect::<Vec<_>>())
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
/// its centre of mass lands on the site position when the site graph has
/// one. Every later copy joins its parent through the port on their bond:
/// the copy is rotated, about its anchor `a`, by the smallest rotation taking
/// its anchor→handle direction `h − a` onto the parent's handle→anchor
/// direction `A − H`, then moved so `a` lands on the parent's handle `H`. The
/// new bond thus points back along the parent's leaving bond, at the length
/// of the parent's anchor–handle bond; bond lengths, ring closures and
/// overlaps are left to a later minimisation.
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
/// let parent = ParentJoin { port, anchor: [0.0; 3], handle: [0.0, 2.0, 0.0] };
/// let site = PlaceSite { position: None, turn: Rigid::IDENTITY, parent: Some(parent) };
/// let pose = GrowthPlacer::new().place(&unit, &site).unwrap();
///
/// // The copy's C lands on the parent's handle, its H points back down.
/// let (c_at, h_at) = (transform_point(&pose, [0.0; 3]), transform_point(&pose, [1.0, 0.0, 0.0]));
/// assert!((c_at[1] - 2.0).abs() < 1e-12 && (h_at[1] - 1.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct GrowthPlacer;

impl GrowthPlacer {
    /// The placer. It takes no configuration.
    pub fn new() -> Self {
        Self
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
    /// template without a centre of mass; [`PlaceError::Port`] when the
    /// parent bond's port does not read back, has an atom without
    /// coordinates, or its anchor and handle coincide.
    fn place(&self, template: &G, site: &PlaceSite) -> Result<Rigid, PlaceError> {
        let Some(join) = site.parent else {
            return match site.position {
                Some(p) => SitePlacer.place(
                    template,
                    &PlaceSite {
                        position: Some(p),
                        ..*site
                    },
                ),
                None => Ok(site.turn),
            };
        };
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
        let shift = Rigid {
            rotation: Rigid::IDENTITY.rotation,
            translation: sub(join.handle, a),
        };
        Ok(compose_rigid(
            &shift,
            &compose_rigid(&turn_about_anchor, &site.turn),
        ))
    }
}

/// Why a [`Placer`] could not place a template.
#[derive(Debug, Clone, PartialEq)]
pub enum PlaceError {
    /// The template has no centre of mass.
    Template(CenterError),
    /// The site has no position and the placer needs one.
    NoPosition,
    /// The site position has a NaN or infinite coordinate.
    NonFinitePoint,
    /// The port joining the parent does not read back or has no direction.
    Port(String),
}

impl fmt::Display for PlaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Template(e) => write!(f, "the template has no centre of mass: {e}"),
            Self::NoPosition => write!(f, "the site has no position"),
            Self::NonFinitePoint => write!(f, "the site position is not finite"),
            Self::Port(why) => write!(f, "the joining port is unusable: {why}"),
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
    use crate::core::CenterError;
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
    fn monomer(with_mass: bool) -> (Atomistic, RelationId, RelationId) {
        let mut m = Atomistic::new();
        let mut ids = Vec::new();
        for (element, x, mass) in [
            (Element::C, 0.0, 12.0),
            (Element::C, 1.5, 12.0),
            (Element::H, -1.0, 1.0),
            (Element::H, 2.5, 1.0),
        ] {
            let id = m
                .add_atom(props! {
                    keys::ATOMIC_NUMBER => element,
                    keys::X => x,
                    keys::Y => 0.0,
                    keys::Z => 0.0,
                })
                .unwrap();
            if with_mass {
                m.set_atom(id, keys::MASS, mass).expect("stamp mass");
            }
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
        let (m, _, _) = monomer(true);
        let pose = SitePlacer::new()
            .place(&m, &site(Some([10.0, 0.0, 0.0]), None))
            .expect("placeable");
        assert_eq!(pose.translation, [9.25, 0.0, 0.0]);
        assert_eq!(pose.rotation, Rigid::IDENTITY.rotation);
    }

    #[test]
    fn site_placer_keeps_the_orienter_turn() {
        let (m, _, _) = monomer(true);
        let rz = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let turned = PlaceSite {
            position: Some([0.0; 3]),
            turn: rotation_about(rz, [0.75, 0.0, 0.0]),
            parent: None,
        };
        let pose = SitePlacer::new().place(&m, &turned).expect("placeable");
        // C1 (1.5,0,0) is 0.75 along +x of the centre: after the quarter turn
        // it sits 0.75 along +y of the site.
        close(transform_point(&pose, [1.5, 0.0, 0.0]), [0.0, 0.75, 0.0]);
    }

    #[test]
    fn site_placer_refuses_a_site_without_a_position_or_a_massless_template() {
        let (m, _, _) = monomer(true);
        assert_eq!(
            SitePlacer::new().place(&m, &site(None, None)),
            Err(PlaceError::NoPosition)
        );
        let (bare, _, _) = monomer(false);
        let err = SitePlacer::new()
            .place(&bare, &site(Some([0.0; 3]), None))
            .expect_err("a massless template has no centre");
        assert!(
            matches!(err, PlaceError::Template(CenterError::BadMass { .. })),
            "{err:?}"
        );
        assert_eq!(
            SitePlacer::new().place(&m, &site(Some([f64::NAN, 0.0, 0.0]), None)),
            Err(PlaceError::NonFinitePoint)
        );
    }

    #[test]
    fn growth_placer_keeps_the_first_copy_where_the_template_is() {
        let (m, _, _) = monomer(false);
        let pose = GrowthPlacer::new()
            .place(&m, &site(None, None))
            .expect("first copy");
        assert_eq!(pose, Rigid::IDENTITY);
    }

    #[test]
    fn growth_placer_puts_the_anchor_on_the_parent_handle_pointing_back() {
        let (m, left, _) = monomer(false);
        // Parent's `>` anchor at (0,0,5), its handle at (0,0,6): the child's
        // `<` anchor C0 lands on (0,0,6) and its handle H0 on the line back,
        // at (0,0,5).
        let join = ParentJoin {
            port: left,
            anchor: [0.0, 0.0, 5.0],
            handle: [0.0, 0.0, 6.0],
        };
        let pose = GrowthPlacer::new()
            .place(&m, &site(None, Some(join)))
            .expect("joined");
        close(transform_point(&pose, [0.0; 3]), [0.0, 0.0, 6.0]);
        close(transform_point(&pose, [-1.0, 0.0, 0.0]), [0.0, 0.0, 5.0]);
    }

    #[test]
    fn growth_placer_refuses_a_join_whose_parent_anchor_and_handle_coincide() {
        let (m, left, _) = monomer(false);
        let join = ParentJoin {
            port: left,
            anchor: [1.0; 3],
            handle: [1.0; 3],
        };
        let err = GrowthPlacer::new()
            .place(&m, &site(None, Some(join)))
            .expect_err("no direction");
        assert!(matches!(err, PlaceError::Port(_)), "{err:?}");
    }
}
