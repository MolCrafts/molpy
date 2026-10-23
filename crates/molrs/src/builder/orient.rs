//! Rotation of template copies: the [`Orienter`] trait and the
//! [`AxisOrienter`].

use std::collections::HashMap;
use std::fmt;

use crate::core::CenterError;
use crate::core::MolGraph;
use crate::core::PortKind;
use crate::core::RelationId;
use crate::core::{Graph, GraphAccess};
use crate::op::vec3::{add, normalize, scale, sub};
use crate::op::{DEFAULT_GAP_TOL, superpose};
use crate::op::{F, Mat3, Vec3};
use crate::op::{Rigid, orthonormal_frame, rotation_about};

/// One bond of a site: the template port the copy joins through and the
/// position of the partner site (Å).
#[derive(Debug, Clone, Copy)]
pub struct SiteLink {
    /// The template port this bond uses.
    pub port: RelationId,
    /// The partner site's position (Å).
    pub toward: Vec3,
}

/// One site to orient: where it sits, its own axis and its bonds.
#[derive(Debug, Clone, Copy)]
pub struct SiteView<'a> {
    /// The site's position (Å).
    pub position: Vec3,
    /// The site's own axis (Å), when the site graph carries one.
    pub axis: Option<Vec3>,
    /// One entry per bond of the site, each with the port it was assigned.
    pub links: &'a [SiteLink],
}

/// Turns one template and a list of sites into one rotation per site.
///
/// An orienter answers the question a [`Placer`](crate::builder::Placer)
/// leaves open (operator, 2026-09-28: the placer only translates, the
/// orienter rotates): which rotation, about the template's centroid,
/// turns a copy before it is moved onto its site? The
/// [`Assembler`](crate::builder::Assembler) holds one of each and applies the
/// orienter's motion first, then the placer's.
///
/// The trait is generic over the template's graph type `G` (a
/// [`MolGraph`], an [`Atomistic`](crate::core::Atomistic) or a
/// [`CoarseGrain`](crate::core::CoarseGrain)), so an implementor reads the
/// template through that type's own vocabulary and an
/// [`Assembler<G>`](crate::builder::Assembler) holds it as a
/// `Box<dyn Orienter<G>>`. [`AxisOrienter`] orients every graph type.
///
/// Implementors are `Send + Sync` so an
/// [`Assembler`](crate::builder::Assembler) holding one can cross threads.
pub trait Orienter<G: Graph>: Send + Sync {
    /// One [`Rigid`] per site of `sites`, in order: a rotation about the
    /// template's centroid (the centre maps to itself), applied to a
    /// copy of `template` before the placer moves it.
    ///
    /// # Errors
    ///
    /// An [`OrientError`] naming why the template or a site cannot be
    /// oriented.
    fn orient_many(&self, template: &G, sites: &[SiteView<'_>]) -> Result<Vec<Rigid>, OrientError>;
}

/// Turns each copy so its own frame matches its site's frame.
///
/// **Chain site** — every bond uses a `<` or `>` port and the template has
/// exactly one of each. The frame is fixed by two directions
/// ([`orthonormal_frame`](crate::op::orthonormal_frame), Gram–Schmidt):
///
/// - site: primary = the site axis; secondary = `Σ ±(q − p)` over its bonds,
///   `+` for the partner `q` on the `>` port and `−` on the `<` port (the
///   central difference `q_> − q_<` along a chain, one-sided at its ends);
/// - template: primary = `R_c − m`, from the midpoint `m` of its `<` and `>`
///   anchors to its centroid `R_c`; secondary = `a_> − a_<`.
///
/// The copy turns by `R = F_site F_templateᵀ` about `R_c`.
///
/// **Branch site** — any other site with bonds (a `$` or `!` port, or a
/// template that is not a two-port chain unit). The rotation is the
/// least-squares fit ([`superpose`](crate::op::superpose)) of the
/// template's port directions (centroid → handle) onto the site's bond
/// directions (site → partner), each set taken with its negation so the fit
/// is a pure rotation about `R_c` (the approach of CG2AT2, Vickery &
/// Stansfeld, *J. Chem. Theory Comput.* **17**, 6472 (2021),
/// doi:10.1021/acs.jctc.1c00295). The site axis is not used.
///
/// A site with no bond is not turned and the template is not read. That the
/// template's size matches the site's is the caller's job.
///
/// # Examples
///
/// ```
/// use molrs::builder::{AxisOrienter, Orienter, SiteLink, SiteView};
/// use molrs::op::transform_point;
/// use molrs::core::keys;
/// use molrs::core::BondNumber;
/// use molrs::core::Atomistic;
/// use molrs::core::PortKind;
/// use molrs::props;
/// use molrs::core::Element;
///
/// // Anchors C0 (−1,0,0) `<` and C1 (1,0,0) `>`, a heavy side atom at
/// // (0,2,0): backbone-to-centre points +y, the joining atoms lie along +x.
/// let mut unit = Atomistic::new();
/// let c0 = unit.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::C,
///     keys::X => -1.0,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// }).unwrap();
/// let c1 = unit.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::C,
///     keys::X => 1.0,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// }).unwrap();
/// let h0 = unit.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::H,
///     keys::X => -2.0,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// }).unwrap();
/// let h1 = unit.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::H,
///     keys::X => 2.0,
///     keys::Y => 0.0,
///     keys::Z => 0.0,
/// }).unwrap();
/// // The side atom is a wildcard dummy (Z = 0): only its mass matters here.
/// let x = unit.add_atom(props! {
///     keys::ATOMIC_NUMBER => 0u8,
///     keys::X => 0.0,
///     keys::Y => 2.0,
///     keys::Z => 0.0,
/// }).unwrap();
/// for (atom, mass) in [(c0, 1.0), (c1, 1.0), (h0, 1.0), (h1, 1.0), (x, 4.0)] {
///     unit.set_atom(atom, keys::MASS, mass).unwrap();
/// }
/// for (a, b) in [(c0, c1), (c0, h0), (c1, h1), (c0, x)] {
///     unit.add_bond(a, b, props! {}).unwrap();
/// }
/// unit.add_port(c0, h0, PortKind::Left, "", BondNumber::Single).unwrap();
/// let right = unit.add_port(c1, h1, PortKind::Right, "", BondNumber::Single).unwrap();
///
/// // A chain end: axis +z, its one partner on the `>` port sits along +y.
/// let links = [SiteLink { port: right, toward: [0.0, 5.0, 0.0] }];
/// let site = SiteView { position: [0.0; 3], axis: Some([0.0, 0.0, 1.0]), links: &links };
/// let r = AxisOrienter::new().orient_many(&unit, &[site]).unwrap();
/// // The joining atoms C0 → C1 now run along +y.
/// let (a, b) = (transform_point(&r[0], [-1.0, 0.0, 0.0]), transform_point(&r[0], [1.0, 0.0, 0.0]));
/// assert!((b[1] - a[1] - 2.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct AxisOrienter;

/// What the orienter reads from a template, once.
struct TemplateGeometry {
    center: Vec3,
    /// Kind and handle position of every port.
    ports: HashMap<RelationId, (PortKind, Vec3)>,
    /// The chain frame, when the template has exactly one `<` and one `>`.
    chain: Option<Mat3>,
}

impl TemplateGeometry {
    fn of(template: &MolGraph) -> Result<Self, OrientError> {
        let position = |atom| -> Result<Vec3, OrientError> {
            template
                .get_node(atom)
                .map_err(|e| OrientError::Template(e.to_string()))?
                .position()
                .ok_or_else(|| OrientError::Template("a port atom has no x/y/z".to_owned()))
        };
        let center = template
            .centroid(&template.node_ids().collect::<Vec<_>>())
            .map_err(OrientError::Center)?;
        let mut ports = HashMap::new();
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for id in template.port_ids() {
            let port = template
                .get_port(id)
                .map_err(|e| OrientError::Template(e.to_string()))?;
            let anchor = position(port.anchor)?;
            match port.kind {
                PortKind::Left => left.push(anchor),
                PortKind::Right => right.push(anchor),
                PortKind::Symmetric | PortKind::Shared => {}
            }
            ports.insert(id, (port.kind, position(port.handle)?));
        }
        let chain = match (left.as_slice(), right.as_slice()) {
            ([l], [r]) => orthonormal_frame(sub(center, scale(add(*l, *r), 0.5)), sub(*r, *l)),
            _ => None,
        };
        Ok(Self {
            center,
            ports,
            chain,
        })
    }
}

impl AxisOrienter {
    /// The orienter. It takes no configuration; each site's axis and bonds
    /// come with the site.
    pub fn new() -> Self {
        Self
    }
}

impl<G: Graph> Orienter<G> for AxisOrienter {
    /// The chain or branch rotation per site (see the type docs);
    /// `Rigid::IDENTITY` for a site with no bond.
    ///
    /// # Errors
    ///
    /// [`OrientError::Template`] / [`OrientError::Center`] when a site needs
    /// the template's geometry and it is missing, or a link names a port the
    /// template lacks; [`OrientError::NoAxis`] for the first chain site
    /// without an axis; [`OrientError::Frame`] for the first site whose
    /// directions fix no rotation.
    fn orient_many(&self, template: &G, sites: &[SiteView<'_>]) -> Result<Vec<Rigid>, OrientError> {
        let mut geometry: Option<TemplateGeometry> = None;
        let mut rigids = Vec::with_capacity(sites.len());
        for (index, site) in sites.iter().enumerate() {
            if site.links.is_empty() {
                rigids.push(Rigid::IDENTITY);
                continue;
            }
            let g = match geometry {
                Some(ref g) => g,
                None => geometry.insert(TemplateGeometry::of(template.graph())?),
            };
            let mut used = Vec::with_capacity(site.links.len());
            for link in site.links {
                let &(kind, handle) = g.ports.get(&link.port).ok_or_else(|| {
                    OrientError::Template(format!(
                        "a link names {:?}, no port of the template",
                        link.port
                    ))
                })?;
                used.push((kind, handle));
            }
            let chain_site = used
                .iter()
                .all(|(k, _)| matches!(k, PortKind::Left | PortKind::Right));
            let rotation = match (chain_site, g.chain) {
                (true, Some(f_template)) => {
                    let axis = site.axis.ok_or(OrientError::NoAxis { index })?;
                    let mut secondary = [0.0; 3];
                    for (link, (kind, _)) in site.links.iter().zip(&used) {
                        let sign = if *kind == PortKind::Right { 1.0 } else { -1.0 };
                        secondary = add(secondary, scale(sub(link.toward, site.position), sign));
                    }
                    let f_site =
                        orthonormal_frame(axis, secondary).ok_or(OrientError::Frame { index })?;
                    mul_transpose(&f_site, &f_template)
                }
                _ => {
                    let reference: Vec<Vec3> =
                        used.iter().map(|(_, h)| sub(*h, g.center)).collect();
                    let target: Vec<Vec3> = site
                        .links
                        .iter()
                        .map(|l| sub(l.toward, site.position))
                        .collect();
                    direction_fit(&reference, &target)
                        .ok_or(OrientError::Frame { index })?
                        .0
                }
            };
            rigids.push(rotation_about(rotation, g.center));
        }
        Ok(rigids)
    }
}

/// `A · Bᵀ`.
fn mul_transpose(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut m = [[0.0; 3]; 3];
    for (r, row) in m.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[r][k] * b[c][k]).sum();
        }
    }
    m
}

/// The rotation best taking the directions of `reference` onto those of
/// `target` (each vector normalised, each set taken with its negation so both
/// centroids are the origin) and the fit's RMSD (dimensionless, on unit
/// vectors). `None` when the sets are empty or differ in length, or a vector
/// is not a direction.
pub(crate) fn direction_fit(reference: &[Vec3], target: &[Vec3]) -> Option<(Mat3, F)> {
    if reference.len() != target.len() || reference.is_empty() {
        return None;
    }
    let mut r = Vec::with_capacity(2 * reference.len());
    let mut t = Vec::with_capacity(2 * target.len());
    for (a, b) in reference.iter().zip(target) {
        let (a, b) = (normalize(*a)?, normalize(*b)?);
        r.extend([a, scale(a, -1.0)]);
        t.extend([b, scale(b, -1.0)]);
    }
    let fit = superpose(&r, &t, &vec![1.0; r.len()], DEFAULT_GAP_TOL).ok()?;
    Some((fit.rigid.rotation, fit.rmsd))
}

/// Why an [`Orienter`] could not orient a template.
#[derive(Debug, Clone, PartialEq)]
pub enum OrientError {
    /// The template's ports do not read back, a port atom has no
    /// coordinates, or a link names a port the template lacks.
    Template(String),
    /// The template has no centroid.
    Center(CenterError),
    /// Chain site `index` has no axis.
    NoAxis {
        /// Index of the offending site in the slice passed.
        index: usize,
    },
    /// Site `index` fixes no rotation: its axis is zero or along its bond
    /// directions, or a bond or port direction is zero.
    Frame {
        /// Index of the offending site in the slice passed.
        index: usize,
    },
}

impl fmt::Display for OrientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Template(why) => write!(f, "the template has no usable geometry: {why}"),
            Self::Center(e) => write!(f, "the template has no centroid: {e}"),
            Self::NoAxis { index } => write!(f, "site {index} is a chain site without an axis"),
            Self::Frame { index } => write!(
                f,
                "site {index} fixes no rotation: its axis is zero or along its bonds, or a direction is zero"
            ),
        }
    }
}

impl std::error::Error for OrientError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Center(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AxisOrienter, OrientError, Orienter, SiteLink, SiteView, direction_fit};
    use crate::core::Atomistic;
    use crate::core::BondNumber;
    use crate::core::Element;
    use crate::core::PortKind;
    use crate::core::RelationId;
    use crate::core::keys;
    use crate::op::Vec3;
    use crate::op::vec3::sub;
    use crate::op::{Rigid, transform_point};
    use crate::props;

    const TOL: f64 = 1e-9;

    /// Chain unit: anchors C0 (−1,0,0) `<` and C1 (1,0,0) `>` with handles at
    /// x = ∓2, and X (0,2,0); masses 1, 1, 1, 1, 4. Centre of mass (0, 1, 0);
    /// chain frame: primary +y, secondary +x. Returns the unit and its
    /// (`<`, `>`) ports.
    fn chain_unit() -> (Atomistic, RelationId, RelationId) {
        let mut f = Atomistic::new();
        let c0 = f
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => -1.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let c1 = f
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 1.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let h0 = f
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::X => -2.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let h1 = f
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::X => 2.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        // The side atom is a wildcard dummy (Z = 0) off the C-C line.
        let x = f
            .add_atom(props! {
                keys::ATOMIC_NUMBER => 0u8,
                keys::X => 0.0,
                keys::Y => 2.0,
                keys::Z => 0.0,
            })
            .unwrap();
        for (a, b) in [(c0, c1), (c0, h0), (c1, h1), (c0, x)] {
            f.add_bond(a, b, props! {}).expect("bond");
        }
        let l = f
            .add_port(c0, h0, PortKind::Left, "", BondNumber::Single)
            .expect("port");
        let r = f
            .add_port(c1, h1, PortKind::Right, "", BondNumber::Single)
            .expect("port");
        (f, l, r)
    }

    /// Branch unit: a centre C at the origin with three `$` hydrogens at
    /// +x, +y and +z (masses 12, 1, 1, 1). Returns it and its ports.
    fn branch_unit() -> (Atomistic, Vec<RelationId>) {
        let mut f = Atomistic::new();
        let c = f
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 0.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        f.set_atom(c, keys::MASS, 12.0).expect("mass");
        let mut ports = Vec::new();
        for d in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            let h = f
                .add_atom(props! {
                    keys::ATOMIC_NUMBER => Element::H,
                    keys::X => d[0],
                    keys::Y => d[1],
                    keys::Z => d[2],
                })
                .unwrap();
            f.set_atom(h, keys::MASS, 1.0).expect("mass");
            f.add_bond(c, h, props! {}).expect("bond");
            ports.push(
                f.add_port(c, h, PortKind::Symmetric, "", BondNumber::Single)
                    .expect("port"),
            );
        }
        (f, ports)
    }

    fn close(got: Vec3, want: Vec3) {
        for d in 0..3 {
            assert!((got[d] - want[d]).abs() < TOL, "want {want:?}, got {got:?}");
        }
    }

    /// The image of a template displacement `from → to` under `r`.
    fn turned(r: &Rigid, from: Vec3, to: Vec3) -> Vec3 {
        sub(transform_point(r, to), transform_point(r, from))
    }

    #[test]
    fn a_chain_site_puts_the_centre_on_its_axis_and_the_joining_atoms_along_its_bonds() {
        let (unit, l, r) = chain_unit();
        // Partner on `<` at (0,−3,0), on `>` at (0,4,0): secondary q_> − q_< = +y.
        let links = [
            SiteLink {
                port: l,
                toward: [0.0, -3.0, 0.0],
            },
            SiteLink {
                port: r,
                toward: [0.0, 4.0, 0.0],
            },
        ];
        let site = SiteView {
            position: [0.0; 3],
            axis: Some([0.0, 0.0, 3.0]),
            links: &links,
        };
        let rot = AxisOrienter::new()
            .orient_many(&unit, &[site])
            .expect("orientable")[0];

        // The centroid of the five atoms (y = 0, 0, 0, 0, 2).
        let centroid = [0.0, 0.4, 0.0];
        close(transform_point(&rot, centroid), centroid);
        close(turned(&rot, [0.0; 3], centroid), [0.0, 0.0, 0.4]);
        close(
            turned(&rot, [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            [0.0, 2.0, 0.0],
        );
    }

    #[test]
    fn a_chain_end_uses_its_one_bond_signed_by_its_port() {
        let (unit, l, _) = chain_unit();
        // Only a `<` partner, at +y: the `>` direction is −y.
        let links = [SiteLink {
            port: l,
            toward: [0.0, 5.0, 0.0],
        }];
        let site = SiteView {
            position: [0.0; 3],
            axis: Some([0.0, 0.0, 1.0]),
            links: &links,
        };
        let rot = AxisOrienter::new()
            .orient_many(&unit, &[site])
            .expect("orientable")[0];

        close(
            turned(&rot, [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            [0.0, -2.0, 0.0],
        );
    }

    #[test]
    fn a_site_without_bonds_is_not_turned_and_the_template_is_not_read() {
        let site = SiteView {
            position: [3.0, 0.0, 0.0],
            axis: None,
            links: &[],
        };
        let rigids = AxisOrienter::new()
            .orient_many(&Atomistic::new(), &[site])
            .expect("nothing to turn");
        assert_eq!(rigids, vec![Rigid::IDENTITY]);
    }

    #[test]
    fn a_chain_site_without_an_axis_is_refused() {
        let (unit, l, _) = chain_unit();
        let links = [SiteLink {
            port: l,
            toward: [0.0, 5.0, 0.0],
        }];
        let site = SiteView {
            position: [0.0; 3],
            axis: None,
            links: &links,
        };
        let err = AxisOrienter::new()
            .orient_many(&unit, &[site])
            .expect_err("no axis");
        assert_eq!(err, OrientError::NoAxis { index: 0 });
    }

    #[test]
    fn a_chain_site_whose_axis_runs_along_its_bond_is_refused() {
        let (unit, l, _) = chain_unit();
        let links = [SiteLink {
            port: l,
            toward: [0.0, 5.0, 0.0],
        }];
        let site = SiteView {
            position: [0.0; 3],
            axis: Some([0.0, 2.0, 0.0]),
            links: &links,
        };
        let err = AxisOrienter::new()
            .orient_many(&unit, &[site])
            .expect_err("degenerate");
        assert_eq!(err, OrientError::Frame { index: 0 });
    }

    #[test]
    fn a_branch_site_turns_each_port_toward_its_partner() {
        let (unit, ports) = branch_unit();
        // Partners along +y, −x, +z: a quarter turn about +z maps
        // +x → +y, +y → −x, +z → +z.
        let links = [
            SiteLink {
                port: ports[0],
                toward: [0.0, 3.0, 0.0],
            },
            SiteLink {
                port: ports[1],
                toward: [-3.0, 0.0, 0.0],
            },
            SiteLink {
                port: ports[2],
                toward: [0.0, 0.0, 3.0],
            },
        ];
        let site = SiteView {
            position: [0.0; 3],
            axis: None,
            links: &links,
        };
        let rot = AxisOrienter::new()
            .orient_many(&unit, &[site])
            .expect("orientable")[0];

        close(turned(&rot, [0.0; 3], [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]);
        close(turned(&rot, [0.0; 3], [0.0, 1.0, 0.0]), [-1.0, 0.0, 0.0]);
    }

    #[test]
    fn direction_fit_is_exact_for_rotated_directions_and_refuses_a_zero_one() {
        let (rot, rmsd) = direction_fit(
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            &[[0.0, 2.0, 0.0], [-3.0, 0.0, 0.0]],
        )
        .expect("a fit");
        assert!(rmsd < TOL, "rmsd {rmsd}");
        close([rot[0][0], rot[1][0], rot[2][0]], [0.0, 1.0, 0.0]);
        assert!(direction_fit(&[[0.0; 3]], &[[1.0, 0.0, 0.0]]).is_none());
    }
}
