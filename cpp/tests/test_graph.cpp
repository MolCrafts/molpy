#include <gtest/gtest.h>
#include "graph.h"

TEST(TestGraph, TestAppendVertex) {
    molpy::Graph<int> graph;
    graph.set_vertex(1, 0);
    ASSERT_EQ(graph.get_num_of_vertices(), 1);
    ASSERT_EQ(graph.get_vertex_label(1), 0);
}

TEST(TestGraph, TestSetEdge) {
    molpy::Graph<int> graph;
    graph.set_edge(0, 1);
    ASSERT_EQ(graph.get_num_of_vertices(), 2);
    graph.set_edge(0, 2);
    ASSERT_EQ(graph.get_num_of_vertices(), 3);
    graph.set_edge(1, 2);
    ASSERT_EQ(graph.get_num_of_vertices(), 3);
}

TEST(TestGraph, TestHasEdge) {
    molpy::Graph<int> graph;
    graph.set_edge(0, 1);
    graph.set_edge(0, 2);
    graph.set_edge(1, 2);

    ASSERT_TRUE(graph.has_edge(0, 1));
    ASSERT_TRUE(graph.has_edge(0, 2));
    ASSERT_TRUE(graph.has_edge(1, 2));
}