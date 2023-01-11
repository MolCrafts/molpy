#include <gtest/gtest.h>
#include "graph.h"

TEST(GraphTest, GraphTest) {
    molpy::Graph<int> graph;
    graph.set_edge(0, 1);
    graph.set_edge(0, 2);
    graph.set_edge(1, 2);

    ASSERT_TRUE(graph.has_edge(0, 1));
    ASSERT_TRUE(graph.has_edge(0, 2));
    ASSERT_TRUE(graph.has_edge(1, 2));
}