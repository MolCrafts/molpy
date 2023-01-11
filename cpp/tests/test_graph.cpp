#include <iostream>
#include "graph.h"

/**
 * Implementation of BFS algorithm is in ../include/graph.h
 */

int main()
{
    std::vector<int> vec{0, 1, 2, 3, 4, 5, 6, 7};
    molpy::Graph<int> g(vec);
    g.setEdge(0, 1);
    g.setEdge(0, 2);
    g.setEdge(1, 0);
    g.setEdge(1, 2);
    g.setEdge(1, 3);
    g.setEdge(2, 4);
    g.setEdge(2, 6);
    g.setEdge(3, 1);
    g.setEdge(3, 7);
    g.setEdge(4, 6);
    g.setEdge(5, 1);
    g.setEdge(5, 3);
    g.setEdge(6, 5);
    g.setEdge(7, 5);
    g.display();
    g.breadth_first_search(0);
    // 0 1 2 3 4 6 7 5 

    molpy::Graph<int> G = molpy::Graph<int>();
    G.setEdge(0, 1);
    G.setEdge(0, 2);
    G.setEdge(1, 0);
    G.setEdge(1, 2);
    G.setEdge(1, 3);
    G.setEdge(2, 4);
    G.setEdge(2, 6);
    G.setEdge(3, 1);
    G.setEdge(3, 7);
    G.setEdge(4, 6);
    G.setEdge(5, 1);
    G.setEdge(5, 3);
    G.setEdge(6, 5);
    G.setEdge(7, 5);
    G.display();
    G.breadth_first_search(0);

    return 0;
}