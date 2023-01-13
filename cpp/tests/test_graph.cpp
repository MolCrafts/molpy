#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "graph.h"

using ::testing::ElementsAreArray;

TEST(TestGraph, TestAppendVertex)
{
    molpy::Graph<int> graph;
    graph.set_vertex(1, 0);
    ASSERT_EQ(graph.get_num_of_vertices(), 1);
    ASSERT_EQ(graph.get_vertex_label(1), 0);
}

TEST(TestGraph, TestSetEdge)
{
    molpy::Graph<int> graph;
    graph.set_edge(0, 1);
    ASSERT_EQ(graph.get_num_of_vertices(), 2);
    graph.set_edge(0, 2);
    ASSERT_EQ(graph.get_num_of_vertices(), 3);
    graph.set_edge(1, 2);
    ASSERT_EQ(graph.get_num_of_vertices(), 3);
}

TEST(TestGraph, TestHasEdge)
{
    molpy::Graph<int> graph;
    graph.set_edge(0, 1);
    graph.set_edge(0, 2);
    graph.set_edge(1, 2);

    ASSERT_TRUE(graph.has_edge(0, 1));
    ASSERT_TRUE(graph.has_edge(0, 2));
    ASSERT_TRUE(graph.has_edge(1, 2));
}

TEST(TestGraph, TestBFS)
{

    molpy::Graph<int> graph;
    graph.set_edge(0, 1);
    graph.set_edge(0, 2);
    graph.set_edge(1, 2);
    graph.set_vertex(3, 3);

    EXPECT_THAT(graph.breadth_first_search(0), ElementsAreArray({0, 1, 2})) << "bfs startswith idx 0 fails";
    EXPECT_THAT(graph.breadth_first_search(1), ElementsAreArray({1, 0, 2})) << "bfs startswith idx 1 fails";
    EXPECT_THAT(graph.breadth_first_search(2), ElementsAreArray({2, 0, 1})) << "bfs startswith idx 2 fails";
    EXPECT_THAT(graph.breadth_first_search(3), ElementsAreArray({3})) << "bfs startswith idx 3 fails";
}

TEST(TestGraph, TestDFS)
{

    molpy::Graph<int> graph;
    graph.set_edge(0, 1);
    graph.set_edge(0, 2);
    graph.set_edge(1, 2);
    graph.set_vertex(3, 3);

    EXPECT_THAT(graph.depth_first_search(0), ElementsAreArray({0, 1, 2})) << "dfs startswith idx 0 fails";
    EXPECT_THAT(graph.depth_first_search(1), ElementsAreArray({1, 0, 2})) << "dfs startswith idx 1 fails";
    EXPECT_THAT(graph.depth_first_search(2), ElementsAreArray({2, 0, 1})) << "dfs startswith idx 2 fails";
    EXPECT_THAT(graph.depth_first_search(3), ElementsAreArray({3})) << "dfs startswith idx 3 fails";
}