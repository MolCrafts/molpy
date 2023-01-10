#ifndef _GRAPH_H
#define _GRAPH_H

#include <vector>
#include <queue>
#include <iostream>

namespace molpy
{

    template <class dataType> // Type of label vertex will hold
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
            dataType label;      // the template label
            Node *list;         // Pointer to all edges (linkedlist)
            int id;  // id of the vertex
        };

        std::vector<Vertex> vertices; // vector of all vertices.

        // private methods
        Node *getNode(Vertex *);                   // allocate and initialize a newnode for the adj list.
        void insertAtEnd(Node *&, Vertex *);       // insert at the end of adjacency list of vertex.
        void deleteAllAfter(Node *);               // delete the adjacency list of the vertex.
        void depth_first_traversal_util(Vertex *); // Private utility function for DFS

    public:
        Graph();                             // = default;                   // Default constructor
        Graph(std::vector<dataType> &);      // Constructor which takes vector of vertex label
        void setEdge(dataType, dataType);   // For setting a edge of graph
        void append_vertex(dataType);
        void display() const;                // Print current config of the graph.
        int get_num_of_vertices() const;     // Get number of vertices in the graph
        void breadth_first_search(dataType); // Breadth first traversal of the graph
        void depth_first_search(dataType);   // Depth first traversal of the graph
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
    Graph<dataType>::Graph(std::vector<dataType> &labels) // Non default constructor, takes a vector of vertices label
        : numOfVertices(labels.size()),
          vertices(numOfVertices)
    {
        for (int i = 0; i < numOfVertices; ++i)
        {
            vertices[i].label = labels[i];
            vertices[i].list = nullptr;
            vertices[i].state = WHITE;
        }
    }

    template <typename dataType>
    void Graph<dataType>::setEdge(dataType data1, dataType data2) // Setting individual edge of the graph.
    {

        if (data1 > numOfVertices - 1 || data2 > numOfVertices - 1)
        {
            int vertices_to_add = data1 > data2 ? data1 : data2;
            for (int i = numOfVertices; i <= vertices_to_add; ++i)
            {
                append_vertex(i);
            }
        }

        for (int i = 0; i < numOfVertices; ++i)
        {
            if (vertices[i].label == data1)
            {
                for (int j = 0; j < numOfVertices; ++j)
                {
                    if (vertices[j].label == data2)
                    {
                        std::cout << "Edge " << vertices[i].label << " -> " << vertices[j].label << " added to the graph" << std::endl;
                        insertAtEnd(vertices[i].list, &vertices[j]);
                        break;
                    }
                }
                break;
            }
        }
    }

    template <typename dataType>
    void Graph<dataType>::append_vertex(dataType value)  // Append empty vertex to the graph
    {
        vertices.push_back(Vertex());
        vertices[numOfVertices].label = value;
        vertices[numOfVertices].list = nullptr;
        vertices[numOfVertices].state = WHITE;
        ++numOfVertices;
        std::cout << "Vertex " << value << " added to the graph" << std::endl;
    }

    template <typename dataType>
    void Graph<dataType>::display() const // Prints the current config of the graph
    {
        Node *node;
        for (int i = 0; i < numOfVertices; ++i)
        {
            std::cout << "Vertex:" << vertices[i].label << " ";
            std::cout << "Connections: ";
            node = vertices[i].list;
            while (node != nullptr)
            {
                std::cout << node->vertexPtr->label << " ";
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
            if (vertices[i].label == startElem)
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
            std::cout << currVertex->label << " " << std::endl;
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
        std::cout << v->label << " ";
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
            if (vertices[i].label == startElem)
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