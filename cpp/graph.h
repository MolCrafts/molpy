#ifndef _GRAPH_H
#define _GRAPH_H

#include <vector>
#include <queue>
#include <iostream>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace molpy
{

    template <class dataType> // Type of idx vertex will hold
    class Graph
    {
        int numOfVertices; // number of vertices.
        struct Vertex;     // forward declaration of vertex structure

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
            dataType label;      // the label of vertex 
            Node *list;         // Pointer to all edges (linkedlist)
            size_t idx;     // index of the vertex
        };

        std::vector<Vertex> vertices; // vector of all vertices.

        // private methods
        Node *getNode(Vertex *);                   // allocate and initialize a newnode for the adj list.
        void insertAtEnd(Node *&, Vertex *);       // insert at the end of adjacency list of vertex.
        void deleteAllAfter(Node *);               // delete the adjacency list of the vertex.
        void depth_first_traversal_util(Vertex *); // Private utility function for DFS

    public:
        Graph();                             // Default constructor
        Graph(std::vector<dataType> &);      // Constructor which takes vector of vertex idx
        Graph(py::array_t<dataType, py::array::c_style | py::array::forcecast> &labels); // Constructor which takes numpy array of vertex idx
        void set_edge(size_t, size_t);   // For setting a edge of graph
        bool has_edge(size_t, size_t);   // For checking if a edge exists
        void append_vertex(dataType);
        void display() const;                // Print current config of the graph.
        int get_num_of_vertices() const;     // Get number of vertices in the graph
        void breadth_first_search(dataType); // Breadth first traversal of the graph
        void depth_first_search(dataType);   // Depth first traversal of the 
        Vertex create_vertex(dataType);    // Create a vertex
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
    Graph<dataType>::Graph() // Default constructor
        : numOfVertices(0),
          vertices(0)
    {
    }

    template <typename dataType>
    Graph<dataType>::Graph(std::vector<dataType> &labels) // takes a vector of vertices labels
        : numOfVertices(labels.size()),
          vertices(numOfVertices)
    {
        for (size_t i = 0; i < numOfVertices; ++i)
        {
            vertices[i].idx = labels[i];
            vertices[i].list = nullptr;
            vertices[i].state = WHITE;
            vertices[i].idx = i;
        }
    }

    template <typename dataType>
    Graph<dataType>::Graph(py::array_t<dataType, py::array::c_style | py::array::forcecast> &labels) // takes a numpy array of vertices labels
        : numOfVertices(labels.size()),
          vertices(numOfVertices)
    {
        for (size_t i = 0; i < numOfVertices; ++i)
        {
            vertices[i].idx = labels.at(i);
            vertices[i].list = nullptr;
            vertices[i].state = WHITE;
            vertices[i].idx = i;
        }
    }

    template <typename dataType>
    void Graph<dataType>::set_edge(size_t idx1, size_t idx2) // Setting individual edge of the graph.
    {

        if (idx1 > numOfVertices - 1 || idx2 > numOfVertices - 1)
        {
            size_t vertices_to_add = idx1 > idx2 ? idx1 : idx2;
            for (size_t i = numOfVertices; i <= vertices_to_add; ++i)
            {
                dataType none_label = std::numeric_limits<dataType>::max();
                append_vertex(none_label);
            }
        }

        for (int i = 0; i < numOfVertices; ++i)
        {
            if (vertices[i].idx == idx1)
            {
                for (int j = 0; j < numOfVertices; ++j)
                {
                    if (vertices[j].idx == idx2)
                    {
                        insertAtEnd(vertices[i].list, &vertices[j]);
                        break;
                    }
                }
                break;
            }
        }
    }

    template <typename dataType>
    bool Graph<dataType>::has_edge(size_t idx1, size_t idx2) // Setting individual edge of the graph.
    {
        for (int i = 0; i < numOfVertices; ++i)
        {
            if (vertices[i].idx == idx1)
            {
                for (int j = 0; j < numOfVertices; ++j)
                {
                    if (vertices[j].idx == idx2)
                    {
                        Node *temp = vertices[i].list;
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
                }
                break;
            }
        }
        return false;
    }


    template <typename dataType>
    void Graph<dataType>::append_vertex(dataType label)  // Append empty vertex to the graph
    {
        vertices.push_back(create_vertex(label));
        ++numOfVertices;
    }

    template <typename dataType>
    Vertex Graph<dataType>::create_vertex(dataType label)
    {
        Vertex v;
        v.label = label;
        v.list = nullptr;
        v.state = WHITE;
        v.index = numOfVertices;
        return v;
    }

    template <typename dataType>
    void Graph<dataType>::display() const // Prints the current config of the graph
    {
        Node *node;
        for (int i = 0; i < numOfVertices; ++i)
        {
            std::cout << "Vertex:" << vertices[i].idx << " ";
            std::cout << "Connections: ";
            node = vertices[i].list;
            while (node != nullptr)
            {
                std::cout << node->vertexPtr->idx << " ";
                node = node->next;
            }
            std::cout << std::endl;
        }
    }

    template <typename dataType>
    int Graph<dataType>::get_num_of_vertices() const // Returns the number of vertices in the graph
    {
        return numOfVertices;
    }

    template <typename dataType>
    void Graph<dataType>::breadth_first_search(dataType startElem) // Breadth first traversal of the graph
    {
        // mark all vertices as not visited, i.e. state = WHITE
        for (int i = 0; i < numOfVertices; ++i)
        {
            vertices[i].state = WHITE;
        }

        // search for the vertex containing start element
        Vertex *startVertex = nullptr;
        for (int i = 0; i < numOfVertices; ++i)
        {
            if (vertices[i].idx == startElem)
            {
                startVertex = &vertices[i];
                break;
            }
        }

        // Return if start vertex not found
        if (startVertex == nullptr)
        {
            return;
        }

        // Create a queue for traversing breadth wise.
        std::queue<Vertex *> vertexQueue;

        // mark the first vertex as being processed
        startVertex->state = GRAY;
        // push the first vertex
        vertexQueue.push(startVertex);
        Vertex *currVertex = nullptr;

        while (!vertexQueue.empty())
        {
            currVertex = vertexQueue.front();
            vertexQueue.pop();
            currVertex->state = BLACK;
            std::cout << currVertex->idx << " " << std::endl;
            Node *adjVertex = currVertex->list;
            while (adjVertex != nullptr)
            {
                if (adjVertex->vertexPtr->state == WHITE)
                {
                    adjVertex->vertexPtr->state = GRAY;
                    vertexQueue.push(adjVertex->vertexPtr);
                }
                adjVertex = adjVertex->next;
            }
        }
        std::cout << std::endl;
    }

    template <typename dataType>
    void Graph<dataType>::depth_first_traversal_util(Vertex *v) // Depth first search private utility function
    {
        v->state = GRAY;
        std::cout << v->idx << " ";
        Node *node = v->list;
        while (node != nullptr)
        {
            if (node->vertexPtr->state == WHITE)
            {
                depth_first_traversal_util(node->vertexPtr);
            }
            node = node->next;
        }
        v->state = BLACK;
    }

    template <typename dataType>
    void Graph<dataType>::depth_first_search(dataType startElem) // Public function for depth first traversal
    {
        for (int i = 0; i < numOfVertices; ++i)
        {
            vertices[i].state = WHITE;
        }
        for (int i = 0; i < numOfVertices; ++i)
        {
            if (vertices[i].idx == startElem)
            {
                depth_first_traversal_util(&vertices[i]);
                break;
            }
        }
        std::cout << std::endl;
    }

    template <typename dataType>
    Graph<dataType>::~Graph()
    {
        for (int i = 0; i < numOfVertices; ++i)
        {
            deleteAllAfter(vertices[i].list);
        }
    }
} // end of namespace molpy

#endif