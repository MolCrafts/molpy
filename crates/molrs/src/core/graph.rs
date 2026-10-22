//! The sealed [`Graph`] trait: the one bound for "any molrs graph type".

use crate::core::MolGraph;
use crate::core::MolRsError;

pub(crate) mod sealed {
    use crate::core::MolGraph;
    use crate::core::MolRsError;

    /// Proof that a call comes from inside molrs. It has no public
    /// constructor, so code outside the crate can name neither it nor the
    /// accessors that take it, even through a `G: Graph` bound.
    pub struct Token(pub(crate) ());

    /// The crate-private half of [`Graph`](super::Graph): access to the
    /// [`MolGraph`] a graph type is built on, and the promotion back.
    pub trait Sealed {
        /// The underlying graph.
        fn inner_graph(&self, token: Token) -> &MolGraph;
        /// The underlying graph, mutably. A typed graph's node invariant
        /// stays installed on it.
        fn inner_graph_mut(&mut self, token: Token) -> &mut MolGraph;
        /// Wrap a finished graph as `Self`, checking `Self`'s invariant.
        fn from_inner_graph(graph: MolGraph, token: Token) -> Result<Self, MolRsError>
        where
            Self: Sized;
    }
}

/// A molrs graph type: [`MolGraph`], [`Atomistic`](crate::core::Atomistic) or
/// [`CoarseGrain`](crate::core::CoarseGrain).
///
/// The trait is sealed — no type outside molrs implements it — and carries
/// no public method: it is the bound an algorithm that works on every graph
/// type takes (`fn f(g: &impl Graph)`), while each type keeps its own typed
/// vocabulary (`add_atom`, `add_bead`, …) as inherent methods.
pub trait Graph: sealed::Sealed {}

/// Crate-internal access to the [`MolGraph`] under any [`Graph`].
///
/// Blanket-implemented for every `G: Graph`; import it where a generic
/// algorithm needs the generic graph (`g.graph()`), or where a builder hands
/// its finished graph back as the caller's type (`G::from_graph(out)`).
pub(crate) trait GraphAccess: Graph {
    /// The underlying graph.
    fn graph(&self) -> &MolGraph {
        self.inner_graph(sealed::Token(()))
    }

    /// The underlying graph, mutably; a typed graph's node invariant stays
    /// in force on it. Test-only: production code mutates a typed graph
    /// through its typed methods, and tests reach past them to prove the
    /// invariant holds on the raw graph too.
    #[cfg(test)]
    fn graph_mut(&mut self) -> &mut MolGraph {
        self.inner_graph_mut(sealed::Token(()))
    }

    /// Wrap `graph` as `Self`, checking `Self`'s invariant.
    ///
    /// # Errors
    ///
    /// The type's own refusal (an atom without a valid atomic number, a bead
    /// without a type, a standard kind registered at a foreign arity).
    fn from_graph(graph: MolGraph) -> Result<Self, MolRsError>
    where
        Self: Sized,
    {
        Self::from_inner_graph(graph, sealed::Token(()))
    }
}

impl<G: Graph> GraphAccess for G {}

impl sealed::Sealed for MolGraph {
    fn inner_graph(&self, _: sealed::Token) -> &MolGraph {
        self
    }

    fn inner_graph_mut(&mut self, _: sealed::Token) -> &mut MolGraph {
        self
    }

    fn from_inner_graph(graph: MolGraph, _: sealed::Token) -> Result<Self, MolRsError> {
        Ok(graph)
    }
}

impl Graph for MolGraph {}
