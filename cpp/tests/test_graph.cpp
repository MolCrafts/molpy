#include <iostream>
#include "graph.h"

/**
 * Implementation of BFS algorithm is in ../include/graph.h
 */

int main()
{
    std::vector<int> vec{0, 1, 2, 3, 4, 5, 6, 7};
    molpy::Graph<int> g(vec);
    g.set_edge(0, 1);
    g.set_edge(0, 2);
    g.set_edge(1, 0);
    g.set_edge(1, 2);
    g.set_edge(1, 3);
    g.set_edge(2, 4);
    g.set_edge(2, 6);
    g.set_edge(3, 1);
    g.set_edge(3, 7);
    g.set_edge(4, 6);
    g.set_edge(5, 1);
    g.set_edge(5, 3);
    g.set_edge(6, 5);
    g.set_edge(7, 5);
    g.display();
    g.breadth_first_search(0);
    return 0;
}