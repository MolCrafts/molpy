#ifndef _GRAPH_H
#define _GRAPH_H

#include <vector>
#include <queue>
#include <map>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

namespace molpy
{
    template <class dataType> // Type of idx vertex will hold
    class Graph
    {
        struct Vertex; // forward declaration of vertex structure

        struct Node
        {                      // linkedlist for mapping edges in the graph
            Vertex *vertexPtr; // points to the vertex to which the edge is adjecent
            Node *next;        // points to the next edge belonging to same vertex
        };

        enum visitedState
        {          // Enum representing visited state of vertex
            WHITE, // not yet visited
            GRAY,  // being visited
            BLACK  // visited
        };
        struct Vertex
        {
            visitedState state; // state of vertex, visited/being visited/done
            dataType label;     // the label of vertex
            size_t idx;         // the index of vertex
            Node *list;         // Pointer to all edges (linkedlist)
        };

        std::map<size_t, Vertex> vertices; // map of all vertices.

        // private methods
        Node *getNode(Vertex *);                                          // allocate and initialize a newnode for the adj list.
        void insertAtEnd(Node *&, Vertex *);                              // insert at the end of adjacency list of vertex.
        void deleteAllAfter(Node *);                                      // delete the adjacency list of the vertex.
        void depth_first_traversal_util(Vertex *, std::vector<size_t> &); // Private utility function for DFS
        Vertex create_vertex(size_t, dataType);                           // Create a vertex
        Vertex get_vertex(size_t);

    public:
        Graph() = default;             // Default constructor
        void set_edge(size_t, size_t); // For setting a edge of graph
        bool has_edge(size_t, size_t); // For checking if a edge exists
        void set_vertex(size_t, dataType);
        bool has_vertex(size_t);
        void display() const;                             // Print current config of the graph.
        int get_num_of_vertices() const;                  // Get number of vertices in the graph
        dataType get_vertex_label(size_t);                // Get the label of the vertex
        std::vector<size_t> breadth_first_search(size_t); // Breadth first traversal of the graph
        std::vector<size_t> depth_first_search(size_t);   // Depth first traversal of the
        ~Graph();
    }; // end of class Graph

    template <typename dataType>
    typename Graph<dataType>::Node *
    molpy::Graph<dataType>::getNode(Vertex *v) // allocate and initialize a newnode for the adj list.
    {
        Node *newNode = new Node;
        newNode->vertexPtr = v;
        newNode->next = nullptr;
        return newNode;
    }

    template <typename dataType>
    dataType Graph<dataType>::get_vertex_label(size_t idx)
    {
        return get_vertex(idx).label;
    }

    template <typename dataType>
    bool Graph<dataType>::has_vertex(size_t idx)
    {
        return vertices.find(idx) != vertices.end();
    }

    template <typename dataType>
    typename Graph<dataType>::Vertex Graph<dataType>::get_vertex(size_t idx)
    {
        auto it = vertices.find(idx);
        if (it != vertices.end())
        {
            return it->second;
        }
        else
        {
            throw py::key_error("Vertex not found");
        }
    }

    template <typename dataType>
    void Graph<dataType>::insertAtEnd(Node *&node, Vertex *v) // insert at the end of adjacency list of vertex.
    {
        Node *newNode = getNode(v);
        if (node == nullptr)
        {
            node = newNode;
        }
        else
        {
            Node *temp = node;
            while (temp->next != nullptr)
            {
                temp = temp->next;
            }
            temp->next = newNode;
        }
    }

    template <typename dataType>
    void Graph<dataType>::deleteAllAfter(Node *node) // delete the adjacency list of the vertex.
    {
        Node *nextNode;
        while (node != nullptr)
        {
            nextNode = node->next;
            delete (node);
            node = nextNode;
        }
    }

    template <typename dataType>
    void Graph<dataType>::set_edge(size_t idx1, size_t idx2) // Setting individual edge of the graph.
    {

        if (!has_vertex(idx1))
            set_vertex(idx1, std::numeric_limits<dataType>::max());
        if (!has_vertex(idx2))
            set_vertex(idx2, std::numeric_limits<dataType>::max());

        insertAtEnd(vertices[idx1].list, &vertices[idx2]);
        insertAtEnd(vertices[idx2].list, &vertices[idx1]);
    }

    template <typename dataType>
    bool Graph<dataType>::has_edge(size_t idx1, size_t idx2) // Setting individual edge of the graph.
    {

        if (!has_vertex(idx1) || !has_vertex(idx2))
            return false;

        Node *temp = vertices[idx1].list;
        while (temp != nullptr)
        {
            if (temp->vertexPtr->idx == idx2)
            {
                return true;
            }
            temp = temp->next;
        }
        return false;
    }

    template <typename dataType>
    void Graph<dataType>::set_vertex(size_t idx, dataType label) // Append empty vertex to the graph
    {
        if (!has_vertex(idx))
        {
            vertices[idx] = create_vertex(idx, label);
        }
    }

    template <typename dataType>
    typename Graph<dataType>::Vertex Graph<dataType>::create_vertex(size_t idx, dataType label)
    {
        Vertex v;
        v.label = label;
        v.list = nullptr;
        v.state = WHITE;
        v.idx = idx;
        return v;
    }

    template <typename dataType>
    void Graph<dataType>::display() const // Prints the current config of the graph
    {
        std::cout << "Graph: " << std::endl;
        for (auto it = vertices.begin(); it != vertices.end(); ++it)
        {
            std::cout << it->first << " : " << it->second.label << " : ";
            Node *temp = it->second.list;
            while (temp != nullptr)
            {
                std::cout << temp->vertexPtr->idx << " ";
                temp = temp->next;
            }
            std::cout << std::endl;
        }
    }

    template <typename dataType>
    int Graph<dataType>::get_num_of_vertices() const // Returns the number of vertices in the graph
    {
        return vertices.size();
    }

    template <typename dataType>
    std::vector<size_t> Graph<dataType>::breadth_first_search(size_t start) // Breadth first traversal of the graph
    {
        // check start vertex exists
        if (!has_vertex(start))
        {
            throw "Vertex not found";
        }
        // mark all vertices as not visited, i.e. state = WHITE
        for (auto it = vertices.begin(); it != vertices.end(); ++it)
        {
            it->second.state = WHITE;
        }

        // create a vector to store vertrices
        std::vector<size_t> bfs_result;

        // create a queue for BFS
        std::queue<Vertex *> q;
        q.push(&vertices[start]);
        vertices[start].state = GRAY;
        while (!q.empty())
        {
            Vertex *v = q.front();
            q.pop();
            // std::cout << v->label << " ";
            bfs_result.push_back(v->idx);
            Node *temp = v->list;
            while (temp != nullptr)
            {
                if (temp->vertexPtr->state == WHITE)
                {
                    temp->vertexPtr->state = GRAY;
                    q.push(temp->vertexPtr);
                }
                temp = temp->next;
            }
            v->state = BLACK;
        }
        return bfs_result;
    }

    template <typename dataType>
    void Graph<dataType>::depth_first_traversal_util(Vertex *v, std::vector<size_t> &result) // Depth first search private utility function
    {
        if (v == nullptr)
        {
            return;
        }
        v->state = GRAY;
        result.push_back(v->idx);
        Node *temp = v->list;
        while (temp != nullptr)
        {
            if (temp->vertexPtr->state == WHITE)
            {
                depth_first_traversal_util(temp->vertexPtr, result);
            }
            temp = temp->next;
        }
        v->state = BLACK;
    }

    template <typename dataType>
    std::vector<size_t> Graph<dataType>::depth_first_search(size_t start) // Public function for depth first traversal
    {

        // check start vertex exists
        if (!has_vertex(start))
        {
            throw "Vertex not found";
        }
        // mark all vertices as not visited, i.e. state = WHITE
        for (auto it = vertices.begin(); it != vertices.end(); ++it)
        {
            it->second.state = WHITE;
        }

        // create a vector to store vertrices
        std::vector<size_t> dfs_result;
        depth_first_traversal_util(&vertices[start], dfs_result);
        return dfs_result;
    }

    template <typename dataType>
    Graph<dataType>::~Graph()
    {
        for (auto it = vertices.begin(); it != vertices.end(); ++it)
        {
            deleteAllAfter(it->second.list);
        }
    }
} // end of namespace molpy

#endif