# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-15
# version: 0.0.1
# source: https://github.com/networkx/networkx/tree/main/networkx/algorithms/isomorphism/
# license: 3-clause BSD

# NOTE: this code is directly copy from networkx/algorithms/isomorphism and slightly modifed to compatible with molpy.graph. For example, nx.Graph.order() is replaced by molpy.graph.order, and nx.Graph.degree() is replaced by molpy.graph.degree and so on. 

"""
 * Copyright (c) 1998, Regents of the University of California
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University of California, Berkeley nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE REGENTS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.}}
"""

"""
*************
VF2 Algorithm
*************

An implementation of VF2 algorithm for graph isomorphism testing.

The simplest interface to use this module is to call networkx.is_isomorphic().

Introduction
------------

The GraphMatcher and DiGraphMatcher are responsible for matching
graphs or directed graphs in a predetermined manner.  This
usually means a check for an isomorphism, though other checks
are also possible.  For example, a subgraph of one graph
can be checked for isomorphism to a second graph.

Matching is done via syntactic feasibility. It is also possible
to check for semantic feasibility. Feasibility, then, is defined
as the logical AND of the two functions.

To include a semantic check, the (Di)GraphMatcher class should be
subclassed, and the semantic_feasibility() function should be
redefined.  By default, the semantic feasibility function always
returns True.  The effect of this is that semantics are not
considered in the matching of G1 and G2.

Examples
--------

Suppose G1 and G2 are isomorphic graphs. Verification is as follows:

>>> from networkx.algorithms import isomorphism
>>> G1 = nx.path_graph(4)
>>> G2 = nx.path_graph(4)
>>> GM = isomorphism.GraphMatcher(G1, G2)
>>> GM.is_isomorphic()
True

GM.mapping stores the isomorphism mapping from G1 to G2.

>>> GM.mapping
{0: 0, 1: 1, 2: 2, 3: 3}


Suppose G1 and G2 are isomorphic directed graphs.
Verification is as follows:

>>> G1 = nx.path_graph(4, create_using=nx.DiGraph())
>>> G2 = nx.path_graph(4, create_using=nx.DiGraph())
>>> DiGM = isomorphism.DiGraphMatcher(G1, G2)
>>> DiGM.is_isomorphic()
True

DiGM.mapping stores the isomorphism mapping from G1 to G2.

>>> DiGM.mapping
{0: 0, 1: 1, 2: 2, 3: 3}



Subgraph Isomorphism
--------------------
Graph theory literature can be ambiguous about the meaning of the
above statement, and we seek to clarify it now.

In the VF2 literature, a mapping M is said to be a graph-subgraph
isomorphism iff M is an isomorphism between G2 and a subgraph of G1.
Thus, to say that G1 and G2 are graph-subgraph isomorphic is to say
that a subgraph of G1 is isomorphic to G2.

Other literature uses the phrase 'subgraph isomorphic' as in 'G1 does
not have a subgraph isomorphic to G2'.  Another use is as an in adverb
for isomorphic.  Thus, to say that G1 and G2 are subgraph isomorphic
is to say that a subgraph of G1 is isomorphic to G2.

Finally, the term 'subgraph' can have multiple meanings. In this
context, 'subgraph' always means a 'node-induced subgraph'. Edge-induced
subgraph isomorphisms are not directly supported, but one should be
able to perform the check by making use of nx.line_graph(). For
subgraphs which are not induced, the term 'monomorphism' is preferred
over 'isomorphism'.

Let G=(N,E) be a graph with a set of nodes N and set of edges E.

If G'=(N',E') is a subgraph, then:
    N' is a subset of N
    E' is a subset of E

If G'=(N',E') is a node-induced subgraph, then:
    N' is a subset of N
    E' is the subset of edges in E relating nodes in N'

If G'=(N',E') is an edge-induced subgraph, then:
    N' is the subset of nodes in N related by edges in E'
    E' is a subset of E

If G'=(N',E') is a monomorphism, then:
    N' is a subset of N
    E' is a subset of the set of edges in E relating nodes in N'

Note that if G' is a node-induced subgraph of G, then it is always a
subgraph monomorphism of G, but the opposite is not always true, as a
monomorphism can have fewer edges.

References
----------
[1]   Luigi P. Cordella, Pasquale Foggia, Carlo Sansone, Mario Vento,
      "A (Sub)Graph Isomorphism Algorithm for Matching Large Graphs",
      IEEE Transactions on Pattern Analysis and Machine Intelligence,
      vol. 26,  no. 10,  pp. 1367-1372,  Oct.,  2004.
      http://ieeexplore.ieee.org/iel5/34/29305/01323804.pdf

[2]   L. P. Cordella, P. Foggia, C. Sansone, M. Vento, "An Improved
      Algorithm for Matching Large Graphs", 3rd IAPR-TC15 Workshop
      on Graph-based Representations in Pattern Recognition, Cuen,
      pp. 149-159, 2001.
      https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.101.5342

See Also
--------
syntactic_feasibility(), semantic_feasibility()

Notes
-----

The implementation handles both directed and undirected graphs as well
as multigraphs.

In general, the subgraph isomorphism problem is NP-complete whereas the
graph isomorphism problem is most likely not NP-complete (although no
polynomial-time algorithm is known to exist).

"""

# This work was originally coded by Christopher Ellison
# as part of the Computational Mechanics Python (CMPy) project.
# James P. Crutchfield, principal investigator.
# Complexity Sciences Center and Physics Department, UC Davis.

import sys


class _GraphMatcher:
    """Implementation of VF2 algorithm for matching undirected graphs.

    Suitable for Graph and MultiGraph instances.
    """

    def __init__(self, G1, G2):
        """Initialize GraphMatcher.

        Parameters
        ----------
        G1,G2: NetworkX Graph or MultiGraph instances.
           The two graphs to check for isomorphism or monomorphism.

        Examples
        --------
        To create a GraphMatcher which checks for syntactic feasibility:

        >>> from networkx.algorithms import isomorphism
        >>> G1 = nx.path_graph(4)
        >>> G2 = nx.path_graph(4)
        >>> GM = isomorphism.GraphMatcher(G1, G2)
        """
        self.G1 = G1
        self.G2 = G2
        self.G1_nodes = set(G1.nodes())
        self.G2_nodes = set(G2.nodes())
        self.G2_node_order = {n: i for i, n in enumerate(G2)}

        # Set recursion limit.
        self.old_recursion_limit = sys.getrecursionlimit()
        expected_max_recursion_level = len(self.G2)
        if self.old_recursion_limit < 1.5 * expected_max_recursion_level:
            # Give some breathing room.
            sys.setrecursionlimit(int(1.5 * expected_max_recursion_level))

        # Declare that we will be searching for a graph-graph isomorphism.
        self.test = "graph"

        # Initialize state
        self.initialize()

    def reset_recursion_limit(self):
        """Restores the recursion limit."""
        # TODO:
        # Currently, we use recursion and set the recursion level higher.
        # It would be nice to restore the level, but because the
        # (Di)GraphMatcher classes make use of cyclic references, garbage
        # collection will never happen when we define __del__() to
        # restore the recursion level. The result is a memory leak.
        # So for now, we do not automatically restore the recursion level,
        # and instead provide a method to do this manually. Eventually,
        # we should turn this into a non-recursive implementation.
        sys.setrecursionlimit(self.old_recursion_limit)

    def candidate_pairs_iter(self):
        """Iterator over candidate pairs of nodes in G1 and G2."""

        # All computations are done using the current state!

        G1_nodes = self.G1_nodes
        G2_nodes = self.G2_nodes
        min_key = self.G2_node_order.__getitem__

        # First we compute the inout-terminal sets.
        T1_inout = [node for node in self.inout_1 if node not in self.core_1]
        T2_inout = [node for node in self.inout_2 if node not in self.core_2]

        # If T1_inout and T2_inout are both nonempty.
        # P(s) = T1_inout x {min T2_inout}
        if T1_inout and T2_inout:
            node_2 = min(T2_inout, key=min_key)
            for node_1 in T1_inout:
                yield node_1, node_2

        else:
            # If T1_inout and T2_inout were both empty....
            # P(s) = (N_1 - M_1) x {min (N_2 - M_2)}
            # if not (T1_inout or T2_inout):  # as suggested by  [2], incorrect
            if 1:  # as inferred from [1], correct
                # First we determine the candidate node for G2
                other_node = min(G2_nodes - set(self.core_2), key=min_key)
                for node in self.G1:
                    if node not in self.core_1:
                        yield node, other_node

        # For all other cases, we don't have any candidate pairs.

    def initialize(self):
        """Reinitializes the state of the algorithm.

        This method should be redefined if using something other than GMState.
        If only subclassing GraphMatcher, a redefinition is not necessary.

        """

        # core_1[n] contains the index of the node paired with n, which is m,
        #           provided n is in the mapping.
        # core_2[m] contains the index of the node paired with m, which is n,
        #           provided m is in the mapping.
        self.core_1 = {}
        self.core_2 = {}

        # See the paper for definitions of M_x and T_x^{y}

        # inout_1[n]  is non-zero if n is in M_1 or in T_1^{inout}
        # inout_2[m]  is non-zero if m is in M_2 or in T_2^{inout}
        #
        # The value stored is the depth of the SSR tree when the node became
        # part of the corresponding set.
        self.inout_1 = {}
        self.inout_2 = {}
        # Practically, these sets simply store the nodes in the subgraph.

        self.state = GMState(self)

        # Provide a convenient way to access the isomorphism mapping.
        self.mapping = self.core_1.copy()

    def is_isomorphic(self):
        """Returns True if G1 and G2 are isomorphic graphs."""

        # Let's do two very quick checks!
        # QUESTION: Should we call faster_graph_could_be_isomorphic(G1,G2)?
        # For now, I just copy the code.

        # Check global properties
        if self.G1.order != self.G2.order: # NOTE: different from networkx
            return False

        # Check local properties
        d1 = sorted(d for d in self.G1.degree.values())
        d2 = sorted(d for d in self.G2.degree.values())
        if d1 != d2:
            return False

        try:
            x = next(self.isomorphisms_iter())
            return True
        except StopIteration:
            return False

    def isomorphisms_iter(self):
        """Generator over isomorphisms between G1 and G2."""
        # Declare that we are looking for a graph-graph isomorphism.
        self.test = "graph"
        self.initialize()
        yield from self.match()

    def match(self):
        """Extends the isomorphism mapping.

        This function is called recursively to determine if a complete
        isomorphism can be found between G1 and G2.  It cleans up the class
        variables after each recursive call. If an isomorphism is found,
        we yield the mapping.

        """
        if len(self.core_1) == len(self.G2):
            # Save the final mapping, otherwise garbage collection deletes it.
            self.mapping = self.core_1.copy()
            # The mapping is complete.
            yield self.mapping
        else:
            for G1_node, G2_node in self.candidate_pairs_iter():
                if self.syntactic_feasibility(G1_node, G2_node):
                    if self.semantic_feasibility(G1_node, G2_node):
                        # Recursive call, adding the feasible state.
                        newstate = self.state.__class__(self, G1_node, G2_node)
                        yield from self.match()

                        # restore data structures
                        newstate.restore()

    def semantic_feasibility(self, G1_node, G2_node):
        """Returns True if adding (G1_node, G2_node) is symantically feasible.

        The semantic feasibility function should return True if it is
        acceptable to add the candidate pair (G1_node, G2_node) to the current
        partial isomorphism mapping.   The logic should focus on semantic
        information contained in the edge data or a formalized node class.

        By acceptable, we mean that the subsequent mapping can still become a
        complete isomorphism mapping.  Thus, if adding the candidate pair
        definitely makes it so that the subsequent mapping cannot become a
        complete isomorphism mapping, then this function must return False.

        The default semantic feasibility function always returns True. The
        effect is that semantics are not considered in the matching of G1
        and G2.

        The semantic checks might differ based on the what type of test is
        being performed.  A keyword description of the test is stored in
        self.test.  Here is a quick description of the currently implemented
        tests::

          test='graph'
            Indicates that the graph matcher is looking for a graph-graph
            isomorphism.

          test='subgraph'
            Indicates that the graph matcher is looking for a subgraph-graph
            isomorphism such that a subgraph of G1 is isomorphic to G2.

          test='mono'
            Indicates that the graph matcher is looking for a subgraph-graph
            monomorphism such that a subgraph of G1 is monomorphic to G2.

        Any subclass which redefines semantic_feasibility() must maintain
        the above form to keep the match() method functional. Implementations
        should consider multigraphs.
        """
        return True

    def subgraph_is_isomorphic(self):
        """Returns True if a subgraph of G1 is isomorphic to G2."""
        try:
            x = next(self.subgraph_isomorphisms_iter())
            return True
        except StopIteration:
            return False

    def subgraph_is_monomorphic(self):
        """Returns True if a subgraph of G1 is monomorphic to G2."""
        try:
            x = next(self.subgraph_monomorphisms_iter())
            return True
        except StopIteration:
            return False

    #    subgraph_is_isomorphic.__doc__ += "\n" + subgraph.replace('\n','\n'+indent)

    def subgraph_isomorphisms_iter(self):
        """Generator over isomorphisms between a subgraph of G1 and G2."""
        # Declare that we are looking for graph-subgraph isomorphism.
        self.test = "subgraph"
        self.initialize()
        yield from self.match()

    def subgraph_monomorphisms_iter(self):
        """Generator over monomorphisms between a subgraph of G1 and G2."""
        # Declare that we are looking for graph-subgraph monomorphism.
        self.test = "mono"
        self.initialize()
        yield from self.match()

    #    subgraph_isomorphisms_iter.__doc__ += "\n" + subgraph.replace('\n','\n'+indent)

    def syntactic_feasibility(self, G1_node, G2_node):
        """Returns True if adding (G1_node, G2_node) is syntactically feasible.

        This function returns True if it is adding the candidate pair
        to the current partial isomorphism/monomorphism mapping is allowable.
        The addition is allowable if the inclusion of the candidate pair does
        not make it impossible for an isomorphism/monomorphism to be found.
        """

        # The VF2 algorithm was designed to work with graphs having, at most,
        # one edge connecting any two nodes.  This is not the case when
        # dealing with an MultiGraphs.
        #
        # Basically, when we test the look-ahead rules R_neighbor, we will
        # make sure that the number of edges are checked. We also add
        # a R_self check to verify that the number of selfloops is acceptable.
        #
        # Users might be comparing Graph instances with MultiGraph instances.
        # So the generic GraphMatcher class must work with MultiGraphs.
        # Care must be taken since the value in the innermost dictionary is a
        # singlet for Graph instances.  For MultiGraphs, the value in the
        # innermost dictionary is a list.

        ###
        # Test at each step to get a return value as soon as possible.
        ###

        # Look ahead 0

        # R_self

        # The number of selfloops for G1_node must equal the number of
        # self-loops for G2_node. Without this check, we would fail on
        # R_neighbor at the next recursion level. But it is good to prune the
        # search tree now.

        if self.test == "mono":
            if self.G1.number_of_edges(G1_node, G1_node) < self.G2.number_of_edges(
                G2_node, G2_node
            ):
                return False
        else:
            if self.G1.number_of_edges(G1_node, G1_node) != self.G2.number_of_edges(
                G2_node, G2_node
            ):
                return False

        # R_neighbor

        # For each neighbor n' of n in the partial mapping, the corresponding
        # node m' is a neighbor of m, and vice versa. Also, the number of
        # edges must be equal.
        if self.test != "mono":
            for neighbor in self.G1[G1_node]:
                if neighbor in self.core_1:
                    if not (self.core_1[neighbor] in self.G2[G2_node]):
                        return False
                    elif self.G1.number_of_edges(
                        neighbor, G1_node
                    ) != self.G2.number_of_edges(self.core_1[neighbor], G2_node):
                        return False

        for neighbor in self.G2[G2_node]:
            if neighbor in self.core_2:
                if not (self.core_2[neighbor] in self.G1[G1_node]):
                    return False
                elif self.test == "mono":
                    if self.G1.number_of_edges(
                        self.core_2[neighbor], G1_node
                    ) < self.G2.number_of_edges(neighbor, G2_node):
                        return False
                else:
                    if self.G1.number_of_edges(
                        self.core_2[neighbor], G1_node
                    ) != self.G2.number_of_edges(neighbor, G2_node):
                        return False

        if self.test != "mono":
            # Look ahead 1

            # R_terminout
            # The number of neighbors of n in T_1^{inout} is equal to the
            # number of neighbors of m that are in T_2^{inout}, and vice versa.
            num1 = 0
            for neighbor in self.G1[G1_node]:
                if (neighbor in self.inout_1) and (neighbor not in self.core_1):
                    num1 += 1
            num2 = 0
            for neighbor in self.G2[G2_node]:
                if (neighbor in self.inout_2) and (neighbor not in self.core_2):
                    num2 += 1
            if self.test == "graph":
                if not (num1 == num2):
                    return False
            else:  # self.test == 'subgraph'
                if not (num1 >= num2):
                    return False

            # Look ahead 2

            # R_new

            # The number of neighbors of n that are neither in the core_1 nor
            # T_1^{inout} is equal to the number of neighbors of m
            # that are neither in core_2 nor T_2^{inout}.
            num1 = 0
            for neighbor in self.G1[G1_node]:
                if neighbor not in self.inout_1:
                    num1 += 1
            num2 = 0
            for neighbor in self.G2[G2_node]:
                if neighbor not in self.inout_2:
                    num2 += 1
            if self.test == "graph":
                if not (num1 == num2):
                    return False
            else:  # self.test == 'subgraph'
                if not (num1 >= num2):
                    return False

        # Otherwise, this node pair is syntactically feasible!
        return True


class _DiGraphMatcher(_GraphMatcher):
    """Implementation of VF2 algorithm for matching directed graphs.

    Suitable for DiGraph and MultiDiGraph instances.
    """

    def __init__(self, G1, G2):
        """Initialize DiGraphMatcher.

        G1 and G2 should be nx.Graph or nx.MultiGraph instances.

        Examples
        --------
        To create a GraphMatcher which checks for syntactic feasibility:

        >>> from networkx.algorithms import isomorphism
        >>> G1 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))
        >>> G2 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))
        >>> DiGM = isomorphism.DiGraphMatcher(G1, G2)
        """
        super().__init__(G1, G2)

    def candidate_pairs_iter(self):
        """Iterator over candidate pairs of nodes in G1 and G2."""

        # All computations are done using the current state!

        G1_nodes = self.G1_nodes
        G2_nodes = self.G2_nodes
        min_key = self.G2_node_order.__getitem__

        # First we compute the out-terminal sets.
        T1_out = [node for node in self.out_1 if node not in self.core_1]
        T2_out = [node for node in self.out_2 if node not in self.core_2]

        # If T1_out and T2_out are both nonempty.
        # P(s) = T1_out x {min T2_out}
        if T1_out and T2_out:
            node_2 = min(T2_out, key=min_key)
            for node_1 in T1_out:
                yield node_1, node_2

        # If T1_out and T2_out were both empty....
        # We compute the in-terminal sets.

        # elif not (T1_out or T2_out):   # as suggested by [2], incorrect
        else:  # as suggested by [1], correct
            T1_in = [node for node in self.in_1 if node not in self.core_1]
            T2_in = [node for node in self.in_2 if node not in self.core_2]

            # If T1_in and T2_in are both nonempty.
            # P(s) = T1_out x {min T2_out}
            if T1_in and T2_in:
                node_2 = min(T2_in, key=min_key)
                for node_1 in T1_in:
                    yield node_1, node_2

            # If all terminal sets are empty...
            # P(s) = (N_1 - M_1) x {min (N_2 - M_2)}

            # elif not (T1_in or T2_in):   # as suggested by  [2], incorrect
            else:  # as inferred from [1], correct
                node_2 = min(G2_nodes - set(self.core_2), key=min_key)
                for node_1 in G1_nodes:
                    if node_1 not in self.core_1:
                        yield node_1, node_2

        # For all other cases, we don't have any candidate pairs.

    def initialize(self):
        """Reinitializes the state of the algorithm.

        This method should be redefined if using something other than DiGMState.
        If only subclassing GraphMatcher, a redefinition is not necessary.
        """

        # core_1[n] contains the index of the node paired with n, which is m,
        #           provided n is in the mapping.
        # core_2[m] contains the index of the node paired with m, which is n,
        #           provided m is in the mapping.
        self.core_1 = {}
        self.core_2 = {}

        # See the paper for definitions of M_x and T_x^{y}

        # in_1[n]  is non-zero if n is in M_1 or in T_1^{in}
        # out_1[n] is non-zero if n is in M_1 or in T_1^{out}
        #
        # in_2[m]  is non-zero if m is in M_2 or in T_2^{in}
        # out_2[m] is non-zero if m is in M_2 or in T_2^{out}
        #
        # The value stored is the depth of the search tree when the node became
        # part of the corresponding set.
        self.in_1 = {}
        self.in_2 = {}
        self.out_1 = {}
        self.out_2 = {}

        self.state = DiGMState(self)

        # Provide a convenient way to access the isomorphism mapping.
        self.mapping = self.core_1.copy()

    def syntactic_feasibility(self, G1_node, G2_node):
        """Returns True if adding (G1_node, G2_node) is syntactically feasible.

        This function returns True if it is adding the candidate pair
        to the current partial isomorphism/monomorphism mapping is allowable.
        The addition is allowable if the inclusion of the candidate pair does
        not make it impossible for an isomorphism/monomorphism to be found.
        """

        # The VF2 algorithm was designed to work with graphs having, at most,
        # one edge connecting any two nodes.  This is not the case when
        # dealing with an MultiGraphs.
        #
        # Basically, when we test the look-ahead rules R_pred and R_succ, we
        # will make sure that the number of edges are checked.  We also add
        # a R_self check to verify that the number of selfloops is acceptable.

        # Users might be comparing DiGraph instances with MultiDiGraph
        # instances. So the generic DiGraphMatcher class must work with
        # MultiDiGraphs. Care must be taken since the value in the innermost
        # dictionary is a singlet for DiGraph instances.  For MultiDiGraphs,
        # the value in the innermost dictionary is a list.

        ###
        # Test at each step to get a return value as soon as possible.
        ###

        # Look ahead 0

        # R_self

        # The number of selfloops for G1_node must equal the number of
        # self-loops for G2_node. Without this check, we would fail on R_pred
        # at the next recursion level. This should prune the tree even further.
        if self.test == "mono":
            if self.G1.number_of_edges(G1_node, G1_node) < self.G2.number_of_edges(
                G2_node, G2_node
            ):
                return False
        else:
            if self.G1.number_of_edges(G1_node, G1_node) != self.G2.number_of_edges(
                G2_node, G2_node
            ):
                return False

        # R_pred

        # For each predecessor n' of n in the partial mapping, the
        # corresponding node m' is a predecessor of m, and vice versa. Also,
        # the number of edges must be equal
        if self.test != "mono":
            for predecessor in self.G1.pred[G1_node]:
                if predecessor in self.core_1:
                    if not (self.core_1[predecessor] in self.G2.pred[G2_node]):
                        return False
                    elif self.G1.number_of_edges(
                        predecessor, G1_node
                    ) != self.G2.number_of_edges(self.core_1[predecessor], G2_node):
                        return False

        for predecessor in self.G2.pred[G2_node]:
            if predecessor in self.core_2:
                if not (self.core_2[predecessor] in self.G1.pred[G1_node]):
                    return False
                elif self.test == "mono":
                    if self.G1.number_of_edges(
                        self.core_2[predecessor], G1_node
                    ) < self.G2.number_of_edges(predecessor, G2_node):
                        return False
                else:
                    if self.G1.number_of_edges(
                        self.core_2[predecessor], G1_node
                    ) != self.G2.number_of_edges(predecessor, G2_node):
                        return False

        # R_succ

        # For each successor n' of n in the partial mapping, the corresponding
        # node m' is a successor of m, and vice versa. Also, the number of
        # edges must be equal.
        if self.test != "mono":
            for successor in self.G1[G1_node]:
                if successor in self.core_1:
                    if not (self.core_1[successor] in self.G2[G2_node]):
                        return False
                    elif self.G1.number_of_edges(
                        G1_node, successor
                    ) != self.G2.number_of_edges(G2_node, self.core_1[successor]):
                        return False

        for successor in self.G2[G2_node]:
            if successor in self.core_2:
                if not (self.core_2[successor] in self.G1[G1_node]):
                    return False
                elif self.test == "mono":
                    if self.G1.number_of_edges(
                        G1_node, self.core_2[successor]
                    ) < self.G2.number_of_edges(G2_node, successor):
                        return False
                else:
                    if self.G1.number_of_edges(
                        G1_node, self.core_2[successor]
                    ) != self.G2.number_of_edges(G2_node, successor):
                        return False

        if self.test != "mono":

            # Look ahead 1

            # R_termin
            # The number of predecessors of n that are in T_1^{in} is equal to the
            # number of predecessors of m that are in T_2^{in}.
            num1 = 0
            for predecessor in self.G1.pred[G1_node]:
                if (predecessor in self.in_1) and (predecessor not in self.core_1):
                    num1 += 1
            num2 = 0
            for predecessor in self.G2.pred[G2_node]:
                if (predecessor in self.in_2) and (predecessor not in self.core_2):
                    num2 += 1
            if self.test == "graph":
                if not (num1 == num2):
                    return False
            else:  # self.test == 'subgraph'
                if not (num1 >= num2):
                    return False

            # The number of successors of n that are in T_1^{in} is equal to the
            # number of successors of m that are in T_2^{in}.
            num1 = 0
            for successor in self.G1[G1_node]:
                if (successor in self.in_1) and (successor not in self.core_1):
                    num1 += 1
            num2 = 0
            for successor in self.G2[G2_node]:
                if (successor in self.in_2) and (successor not in self.core_2):
                    num2 += 1
            if self.test == "graph":
                if not (num1 == num2):
                    return False
            else:  # self.test == 'subgraph'
                if not (num1 >= num2):
                    return False

            # R_termout

            # The number of predecessors of n that are in T_1^{out} is equal to the
            # number of predecessors of m that are in T_2^{out}.
            num1 = 0
            for predecessor in self.G1.pred[G1_node]:
                if (predecessor in self.out_1) and (predecessor not in self.core_1):
                    num1 += 1
            num2 = 0
            for predecessor in self.G2.pred[G2_node]:
                if (predecessor in self.out_2) and (predecessor not in self.core_2):
                    num2 += 1
            if self.test == "graph":
                if not (num1 == num2):
                    return False
            else:  # self.test == 'subgraph'
                if not (num1 >= num2):
                    return False

            # The number of successors of n that are in T_1^{out} is equal to the
            # number of successors of m that are in T_2^{out}.
            num1 = 0
            for successor in self.G1[G1_node]:
                if (successor in self.out_1) and (successor not in self.core_1):
                    num1 += 1
            num2 = 0
            for successor in self.G2[G2_node]:
                if (successor in self.out_2) and (successor not in self.core_2):
                    num2 += 1
            if self.test == "graph":
                if not (num1 == num2):
                    return False
            else:  # self.test == 'subgraph'
                if not (num1 >= num2):
                    return False

            # Look ahead 2

            # R_new

            # The number of predecessors of n that are neither in the core_1 nor
            # T_1^{in} nor T_1^{out} is equal to the number of predecessors of m
            # that are neither in core_2 nor T_2^{in} nor T_2^{out}.
            num1 = 0
            for predecessor in self.G1.pred[G1_node]:
                if (predecessor not in self.in_1) and (predecessor not in self.out_1):
                    num1 += 1
            num2 = 0
            for predecessor in self.G2.pred[G2_node]:
                if (predecessor not in self.in_2) and (predecessor not in self.out_2):
                    num2 += 1
            if self.test == "graph":
                if not (num1 == num2):
                    return False
            else:  # self.test == 'subgraph'
                if not (num1 >= num2):
                    return False

            # The number of successors of n that are neither in the core_1 nor
            # T_1^{in} nor T_1^{out} is equal to the number of successors of m
            # that are neither in core_2 nor T_2^{in} nor T_2^{out}.
            num1 = 0
            for successor in self.G1[G1_node]:
                if (successor not in self.in_1) and (successor not in self.out_1):
                    num1 += 1
            num2 = 0
            for successor in self.G2[G2_node]:
                if (successor not in self.in_2) and (successor not in self.out_2):
                    num2 += 1
            if self.test == "graph":
                if not (num1 == num2):
                    return False
            else:  # self.test == 'subgraph'
                if not (num1 >= num2):
                    return False

        # Otherwise, this node pair is syntactically feasible!
        return True


class GMState:
    """Internal representation of state for the GraphMatcher class.

    This class is used internally by the GraphMatcher class.  It is used
    only to store state specific data. There will be at most G2.order of
    these objects in memory at a time, due to the depth-first search
    strategy employed by the VF2 algorithm.
    """

    def __init__(self, GM, G1_node=None, G2_node=None):
        """Initializes GMState object.

        Pass in the GraphMatcher to which this GMState belongs and the
        new node pair that will be added to the GraphMatcher's current
        isomorphism mapping.
        """
        self.GM = GM

        # Initialize the last stored node pair.
        self.G1_node = None
        self.G2_node = None
        self.depth = len(GM.core_1)

        if G1_node is None or G2_node is None:
            # Then we reset the class variables
            GM.core_1 = {}
            GM.core_2 = {}
            GM.inout_1 = {}
            GM.inout_2 = {}

        # Watch out! G1_node == 0 should evaluate to True.
        if G1_node is not None and G2_node is not None:
            # Add the node pair to the isomorphism mapping.
            GM.core_1[G1_node] = G2_node
            GM.core_2[G2_node] = G1_node

            # Store the node that was added last.
            self.G1_node = G1_node
            self.G2_node = G2_node

            # Now we must update the other two vectors.
            # We will add only if it is not in there already!
            self.depth = len(GM.core_1)

            # First we add the new nodes...
            if G1_node not in GM.inout_1:
                GM.inout_1[G1_node] = self.depth
            if G2_node not in GM.inout_2:
                GM.inout_2[G2_node] = self.depth

            # Now we add every other node...

            # Updates for T_1^{inout}
            new_nodes = set()
            for node in GM.core_1:
                new_nodes.update(
                    [neighbor for neighbor in GM.G1[node] if neighbor not in GM.core_1]
                )
            for node in new_nodes:
                if node not in GM.inout_1:
                    GM.inout_1[node] = self.depth

            # Updates for T_2^{inout}
            new_nodes = set()
            for node in GM.core_2:
                new_nodes.update(
                    [neighbor for neighbor in GM.G2[node] if neighbor not in GM.core_2]
                )
            for node in new_nodes:
                if node not in GM.inout_2:
                    GM.inout_2[node] = self.depth

    def restore(self):
        """Deletes the GMState object and restores the class variables."""
        # First we remove the node that was added from the core vectors.
        # Watch out! G1_node == 0 should evaluate to True.
        if self.G1_node is not None and self.G2_node is not None:
            del self.GM.core_1[self.G1_node]
            del self.GM.core_2[self.G2_node]

        # Now we revert the other two vectors.
        # Thus, we delete all entries which have this depth level.
        for vector in (self.GM.inout_1, self.GM.inout_2):
            for node in list(vector.keys()):
                if vector[node] == self.depth:
                    del vector[node]


class DiGMState:
    """Internal representation of state for the DiGraphMatcher class.

    This class is used internally by the DiGraphMatcher class.  It is used
    only to store state specific data. There will be at most G2.order of
    these objects in memory at a time, due to the depth-first search
    strategy employed by the VF2 algorithm.

    """

    def __init__(self, GM, G1_node=None, G2_node=None):
        """Initializes DiGMState object.

        Pass in the DiGraphMatcher to which this DiGMState belongs and the
        new node pair that will be added to the GraphMatcher's current
        isomorphism mapping.
        """
        self.GM = GM

        # Initialize the last stored node pair.
        self.G1_node = None
        self.G2_node = None
        self.depth = len(GM.core_1)

        if G1_node is None or G2_node is None:
            # Then we reset the class variables
            GM.core_1 = {}
            GM.core_2 = {}
            GM.in_1 = {}
            GM.in_2 = {}
            GM.out_1 = {}
            GM.out_2 = {}

        # Watch out! G1_node == 0 should evaluate to True.
        if G1_node is not None and G2_node is not None:
            # Add the node pair to the isomorphism mapping.
            GM.core_1[G1_node] = G2_node
            GM.core_2[G2_node] = G1_node

            # Store the node that was added last.
            self.G1_node = G1_node
            self.G2_node = G2_node

            # Now we must update the other four vectors.
            # We will add only if it is not in there already!
            self.depth = len(GM.core_1)

            # First we add the new nodes...
            for vector in (GM.in_1, GM.out_1):
                if G1_node not in vector:
                    vector[G1_node] = self.depth
            for vector in (GM.in_2, GM.out_2):
                if G2_node not in vector:
                    vector[G2_node] = self.depth

            # Now we add every other node...

            # Updates for T_1^{in}
            new_nodes = set()
            for node in GM.core_1:
                new_nodes.update(
                    [
                        predecessor
                        for predecessor in GM.G1.predecessors(node)
                        if predecessor not in GM.core_1
                    ]
                )
            for node in new_nodes:
                if node not in GM.in_1:
                    GM.in_1[node] = self.depth

            # Updates for T_2^{in}
            new_nodes = set()
            for node in GM.core_2:
                new_nodes.update(
                    [
                        predecessor
                        for predecessor in GM.G2.predecessors(node)
                        if predecessor not in GM.core_2
                    ]
                )
            for node in new_nodes:
                if node not in GM.in_2:
                    GM.in_2[node] = self.depth

            # Updates for T_1^{out}
            new_nodes = set()
            for node in GM.core_1:
                new_nodes.update(
                    [
                        successor
                        for successor in GM.G1.successors(node)
                        if successor not in GM.core_1
                    ]
                )
            for node in new_nodes:
                if node not in GM.out_1:
                    GM.out_1[node] = self.depth

            # Updates for T_2^{out}
            new_nodes = set()
            for node in GM.core_2:
                new_nodes.update(
                    [
                        successor
                        for successor in GM.G2.successors(node)
                        if successor not in GM.core_2
                    ]
                )
            for node in new_nodes:
                if node not in GM.out_2:
                    GM.out_2[node] = self.depth

    def restore(self):
        """Deletes the DiGMState object and restores the class variables."""

        # First we remove the node that was added from the core vectors.
        # Watch out! G1_node == 0 should evaluate to True.
        if self.G1_node is not None and self.G2_node is not None:
            del self.GM.core_1[self.G1_node]
            del self.GM.core_2[self.G2_node]

        # Now we revert the other four vectors.
        # Thus, we delete all entries which have this depth level.
        for vector in (self.GM.in_1, self.GM.in_2, self.GM.out_1, self.GM.out_2):
            for node in list(vector.keys()):
                if vector[node] == self.depth:
                    del vector[node]


"""
    Module to simplify the specification of user-defined equality functions for
    node and edge attributes during isomorphism checks.

    During the construction of an isomorphism, the algorithm considers two
    candidate nodes n1 in G1 and n2 in G2.  The graphs G1 and G2 are then
    compared with respect to properties involving n1 and n2, and if the outcome
    is good, then the candidate nodes are considered isomorphic. NetworkX
    provides a simple mechanism for users to extend the comparisons to include
    node and edge attributes.

    Node attributes are handled by the node_match keyword. When considering
    n1 and n2, the algorithm passes their node attribute dictionaries to
    node_match, and if it returns False, then n1 and n2 cannot be
    considered to be isomorphic.

    Edge attributes are handled by the edge_match keyword. When considering
    n1 and n2, the algorithm must verify that outgoing edges from n1 are
    commensurate with the outgoing edges for n2. If the graph is directed,
    then a similar check is also performed for incoming edges.

    Focusing only on outgoing edges, we consider pairs of nodes (n1, v1) from
    G1 and (n2, v2) from G2. For graphs and digraphs, there is only one edge
    between (n1, v1) and only one edge between (n2, v2). Those edge attribute
    dictionaries are passed to edge_match, and if it returns False, then
    n1 and n2 cannot be considered isomorphic. For multigraphs and
    multidigraphs, there can be multiple edges between (n1, v1) and also
    multiple edges between (n2, v2).  Now, there must exist an isomorphism
    from "all the edges between (n1, v1)" to "all the edges between (n2, v2)".
    So, all of the edge attribute dictionaries are passed to edge_match, and
    it must determine if there is an isomorphism between the two sets of edges.
"""

def _semantic_feasibility(self, G1_node, G2_node):
    """Returns True if mapping G1_node to G2_node is semantically feasible."""
    # Make sure the nodes match
    if self.node_match is not None:
        nm = self.node_match(self.G1.nodes[G1_node], self.G2.nodes[G2_node])
        if not nm:
            return False

    # Make sure the edges match
    if self.edge_match is not None:

        # Cached lookups
        G1nbrs = self.G1_adj[G1_node]
        G2nbrs = self.G2_adj[G2_node]
        core_1 = self.core_1
        edge_match = self.edge_match

        for neighbor in G1nbrs:
            # G1_node is not in core_1, so we must handle R_self separately
            if neighbor == G1_node:
                if G2_node in G2nbrs and not edge_match(
                    G1nbrs[G1_node], G2nbrs[G2_node]
                ):
                    return False
            elif neighbor in core_1:
                G2_nbr = core_1[neighbor]
                if G2_nbr in G2nbrs and not edge_match(
                    G1nbrs[neighbor], G2nbrs[G2_nbr]
                ):
                    return False
        # syntactic check has already verified that neighbors are symmetric

    return True


class GraphMatcher(_GraphMatcher):
    """VF2 isomorphism checker for undirected graphs."""

    def __init__(self, G1, G2, node_match=None, edge_match=None):
        """Initialize graph matcher.

        Parameters
        ----------
        G1, G2: graph
            The graphs to be tested.

        node_match: callable
            A function that returns True iff node n1 in G1 and n2 in G2
            should be considered equal during the isomorphism test. The
            function will be called like::

               node_match(G1.nodes[n1], G2.nodes[n2])

            That is, the function will receive the node attribute dictionaries
            of the nodes under consideration. If None, then no attributes are
            considered when testing for an isomorphism.

        edge_match: callable
            A function that returns True iff the edge attribute dictionary for
            the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should be
            considered equal during the isomorphism test. The function will be
            called like::

               edge_match(G1[u1][v1], G2[u2][v2])

            That is, the function will receive the edge attribute dictionaries
            of the edges under consideration. If None, then no attributes are
            considered when testing for an isomorphism.

        """
        _GraphMatcher.__init__(self, G1, G2)

        self.node_match = node_match
        self.edge_match = edge_match

        # These will be modified during checks to minimize code repeat.
        self.G1_adj = self.G1.adj
        self.G2_adj = self.G2.adj

    semantic_feasibility = _semantic_feasibility


class DiGraphMatcher(_DiGraphMatcher):
    """VF2 isomorphism checker for directed graphs."""

    def __init__(self, G1, G2, node_match=None, edge_match=None):
        """Initialize graph matcher.

        Parameters
        ----------
        G1, G2 : graph
            The graphs to be tested.

        node_match : callable
            A function that returns True iff node n1 in G1 and n2 in G2
            should be considered equal during the isomorphism test. The
            function will be called like::

               node_match(G1.nodes[n1], G2.nodes[n2])

            That is, the function will receive the node attribute dictionaries
            of the nodes under consideration. If None, then no attributes are
            considered when testing for an isomorphism.

        edge_match : callable
            A function that returns True iff the edge attribute dictionary for
            the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should be
            considered equal during the isomorphism test. The function will be
            called like::

               edge_match(G1[u1][v1], G2[u2][v2])

            That is, the function will receive the edge attribute dictionaries
            of the edges under consideration. If None, then no attributes are
            considered when testing for an isomorphism.

        """
        _DiGraphMatcher.__init__(self, G1, G2)

        self.node_match = node_match
        self.edge_match = edge_match

        # These will be modified during checks to minimize code repeat.
        self.G1_adj = self.G1.adj
        self.G2_adj = self.G2.adj

    def semantic_feasibility(self, G1_node, G2_node):
        """Returns True if mapping G1_node to G2_node is semantically feasible."""

        # Test node_match and also test edge_match on successors
        feasible = _semantic_feasibility(self, G1_node, G2_node)
        if not feasible:
            return False

        # Test edge_match on predecessors
        self.G1_adj = self.G1.pred
        self.G2_adj = self.G2.pred
        feasible = _semantic_feasibility(self, G1_node, G2_node)
        self.G1_adj = self.G1.adj
        self.G2_adj = self.G2.adj

        return feasible


# The "semantics" of edge_match are different for multi(di)graphs, but
# the implementation is the same.  So, technically we do not need to
# provide "multi" versions, but we do so to match NetworkX's base classes.


class MultiGraphMatcher(GraphMatcher):
    """VF2 isomorphism checker for undirected multigraphs."""

    pass


class MultiDiGraphMatcher(DiGraphMatcher):
    """VF2 isomorphism checker for directed multigraphs."""

    pass


def faster_could_be_isomorphic(G1, G2):
    """Returns False if graphs are definitely not isomorphic.

    True does NOT guarantee isomorphism.

    Parameters
    ----------
    G1, G2 : graphs
       The two graphs G1 and G2 must be the same type.

    Notes
    -----
    Checks for matching degree sequences.
    """
    # Check global properties
    if G1.order != G2.order:
        return False

    # Check local properties
    d1 = sorted(d for d in G1.degree.values())
    d2 = sorted(d for d in G2.degree.values())

    if d1 != d2:
        return False

    # OK...
    return True


faster_graph_could_be_isomorphic = faster_could_be_isomorphic


def is_isomorphic(G1, G2, node_match=None, edge_match=None):
    """Returns True if the graphs G1 and G2 are isomorphic and False otherwise.

    Parameters
    ----------
    G1, G2: graphs
        The two graphs G1 and G2 must be the same type.

    node_match : callable
        A function that returns True if node n1 in G1 and n2 in G2 should
        be considered equal during the isomorphism test.
        If node_match is not specified then node attributes are not considered.

        The function will be called like

           node_match(G1.nodes[n1], G2.nodes[n2]).

        That is, the function will receive the node attribute dictionaries
        for n1 and n2 as inputs.

    edge_match : callable
        A function that returns True if the edge attribute dictionary
        for the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should
        be considered equal during the isomorphism test.  If edge_match is
        not specified then edge attributes are not considered.

        The function will be called like

           edge_match(G1[u1][v1], G2[u2][v2]).

        That is, the function will receive the edge attribute dictionaries
        of the edges under consideration.

    Notes
    -----
    Uses the vf2 algorithm [1]_.

    Examples
    --------
    >>> import networkx.algorithms.isomorphism as iso

    For digraphs G1 and G2, using 'weight' edge attribute (default: 1)

    >>> G1 = nx.DiGraph()
    >>> G2 = nx.DiGraph()
    >>> nx.add_path(G1, [1, 2, 3, 4], weight=1)
    >>> nx.add_path(G2, [10, 20, 30, 40], weight=2)
    >>> em = iso.numerical_edge_match("weight", 1)
    >>> nx.is_isomorphic(G1, G2)  # no weights considered
    True
    >>> nx.is_isomorphic(G1, G2, edge_match=em)  # match weights
    False

    For multidigraphs G1 and G2, using 'fill' node attribute (default: '')

    >>> G1 = nx.MultiDiGraph()
    >>> G2 = nx.MultiDiGraph()
    >>> G1.add_nodes_from([1, 2, 3], fill="red")
    >>> G2.add_nodes_from([10, 20, 30, 40], fill="red")
    >>> nx.add_path(G1, [1, 2, 3, 4], weight=3, linewidth=2.5)
    >>> nx.add_path(G2, [10, 20, 30, 40], weight=3)
    >>> nm = iso.categorical_node_match("fill", "red")
    >>> nx.is_isomorphic(G1, G2, node_match=nm)
    True

    For multidigraphs G1 and G2, using 'weight' edge attribute (default: 7)

    >>> G1.add_edge(1, 2, weight=7)
    1
    >>> G2.add_edge(10, 20)
    1
    >>> em = iso.numerical_multiedge_match("weight", 7, rtol=1e-6)
    >>> nx.is_isomorphic(G1, G2, edge_match=em)
    True

    For multigraphs G1 and G2, using 'weight' and 'linewidth' edge attributes
    with default values 7 and 2.5. Also using 'fill' node attribute with
    default value 'red'.

    >>> em = iso.numerical_multiedge_match(["weight", "linewidth"], [7, 2.5])
    >>> nm = iso.categorical_node_match("fill", "red")
    >>> nx.is_isomorphic(G1, G2, edge_match=em, node_match=nm)
    True

    See Also
    --------
    numerical_node_match, numerical_edge_match, numerical_multiedge_match
    categorical_node_match, categorical_edge_match, categorical_multiedge_match

    References
    ----------
    .. [1]  L. P. Cordella, P. Foggia, C. Sansone, M. Vento,
       "An Improved Algorithm for Matching Large Graphs",
       3rd IAPR-TC15 Workshop  on Graph-based Representations in
       Pattern Recognition, Cuen, pp. 149-159, 2001.
       https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.101.5342
    """
    if G1.is_directed and G2.is_directed:
        GM = DiGraphMatcher
    elif (not G1.is_directed) and (not G2.is_directed):
        GM = GraphMatcher
    else:
        raise TypeError("Graphs G1 and G2 are not of the same type.")

    gm = GM(G1, G2, node_match=node_match, edge_match=edge_match)

    return gm.is_isomorphic()
