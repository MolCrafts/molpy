// #include <iostream>
// #include <fstream>
// #include <string>
// #include <vector>
// #include <map>
// #include <set>
// #include <algorithm>
// #include <graph.h>
// using namespace std;

// vector<vector<int>>
// dijkstra(vector<vector<int>> graph);
// vector<vector<int>> reindex(vector<vector<int>> graph, vector<int> index);
// vector<int> inv(vector<int> index);
// vector<int> transform(map<vector<int>, vector<int>> signmatrixA, map<vector<int>, vector<int>> signmatrixB, vector<int> vertexA, vector<int> vertexB, vector<int> isoB);
// ifstream infileA("graphA.txt");
// ifstream infileB("graphB.txt");
// ofstream outfile("result.txt");

// namespace molpy
// {
//     /template<typename dataType>
//     bool faster_could_be_isomorphic(Graph<dataType> graphA, Graph<dataType> graphB)
//     {
//         if (graphA.getVertexNum() != graphB.getVertexNum())
//             return false;

//         vector<int> graphA_degree;
//         vector<int> graphB_degree;

//         for (int i = 0; i < graphA.getVertexNum(); i++)
//         {
//             graphA_degree.push_back(graphA.getDegree(i));
//             graphB_degree.push_back(graphB.getDegree(i));
//         }

//         if (graphA_degree.size() != graphB_degree.size())
//             return false;

//         sort(graphA_degree.begin(), graphA_degree.end());
//         sort(graphB_degree.begin(), graphB_degree.end());

//         if (graphA_degree != graphB_degree)
//             return false;

//         return true;

//     }

// }

// template<typename dataType>
// Graph<dataType>::isomorphism(Graph<dataType> graph)
// {

// }

// int main()
// {
//     cout << "The Graph Isomorphism Algorithm" << endl;
//     cout << "by Ashay Dharwadker and John-Tagore Tevet" << endl;
//     cout << "http://www.dharwadker.org/tevet/isomorphism/" << endl;
//     cout << "Copyright (c) 2009" << endl;

//     // Process Graph A
//     cout
//         << "Computing the Sign Matrix of Graph A..." << endl;
//     int i, j, k, q, N1, N2, nA;
//     infileA >> nA;

//     // Read adjacency matrix of graph A from graphA.txt
//     vector<vector<int>>
//         rgraphA;
//     int valA;
//     for (i = 0; i < nA; i++)
//     {
//         vector<int> rowA;
//         for (j = 0; j < nA; j++)
//         {
//             infileA >> valA;
//             rowA.push_back(valA);
//         }
//         rgraphA.push_back(rowA);
//     }

//     // End reading

//     // Initial sorting
//     vector<vector<int>>
//         distanceA;
//     vector<vector<int>> neighborA;
//     for (i = 0; i < rgraphA.size(); i++)
//     {
//         vector<int> rowdistanceA, rowneighborA;
//         for (j = 0; j < rgraphA[i].size(); j++)
//         {
//             if (rgraphA[i][j] == 1)
//                 rowneighborA.push_back(j);
//             vector<int> indexdA;
//             for (k = 0; k < rgraphA.size(); k++)
//                 indexdA.push_back(k);
//             indexdA[0] = i;
//             indexdA[i] = 0;
//             vector<vector<int>> dgraphA = reindex(rgraphA, indexdA);
//             vector<vector<int>> dpathA = dijkstra(dgraphA);
//             int ddA = -1;
//             for (k = 0; k < dpathA.size(); k++)
//                 if (inv(indexdA)[dpathA[k][dpathA[k].size() - 1]] == j)
//                 {
//                     ddA = dpathA[k].size() - 1;
//                     break;
//                 }
//             rowdistanceA.push_back(ddA);
//         }
//         distanceA.push_back(rowdistanceA);
//         neighborA.push_back(rowneighborA);
//     }

//     vector<vector<int>>
//         sequenceA;
//     for (i = 0; i < rgraphA.size(); i++)
//     {
//         vector<int> mutualdistanceA;
//         for (j = 0; j < neighborA[i].size() - 1; j++)
//             for (k = j + 1; k < neighborA[i].size(); k++)
//                 mutualdistanceA.push_back(distanceA[neighborA[i][j]][neighborA[i][k]]);
//         sort(mutualdistanceA.begin(), mutualdistanceA.end());
//         vector<int> tmutualA;
//         for (k = 0; k < nA * nA - 1 - mutualdistanceA.size(); k++)
//             tmutualA.push_back(0);
//         for (k = 0; k < mutualdistanceA.size(); k++)
//             tmutualA.push_back(mutualdistanceA[k]);
//         sequenceA.push_back(tmutualA);
//     }

//     map<vector<int>, int>
//         sorterA;
//     for (i = 0; i < sequenceA.size(); i++)
//         sorterA[sequenceA[i]] = i;
//     vector<int> mainindexA;
//     map<vector<int>, int>::iterator sitA = sorterA.begin();
//     while (sitA != sorterA.end())
//     {
//         vector<int> vsitA = sitA->first;
//         for (i = 0; i < sequenceA.size(); i++)
//             if (sequenceA[i] == vsitA)
//                 mainindexA.push_back(i);
//         sitA++;
//     }

//     vector<vector<int>>
//         graphA = reindex(rgraphA, mainindexA);

//     // Compute degree sequence
//     vector<int>
//         degA;
//     int sumA;
//     for (i = 0; i < nA; i++)
//     {
//         sumA = 0;
//         for (j = 0; j < nA; j++)
//         {
//             if (graphA[i][j] == 1)
//                 sumA++;
//         }
//         degA.push_back(sumA);
//     }
//     vector<int> sorted_degA = degA;
//     sort(sorted_degA.begin(), sorted_degA.end());

//     // Pair graphs
//     map<vector<int>, set<int>>
//         PA;
//     map<vector<int>, int> dA;
//     map<vector<int>, int> ntA;
//     map<vector<int>, int> eA;

//     for (N1 = 0; N1 < graphA.size(); N1++)
//         for (N2 = 0; N2 < graphA[N1].size(); N2++)
//         {
//             int signA = -1;
//             vector<vector<int>> tgraphA = graphA;
//             if (graphA[N1][N2] != 0)
//             {
//                 tgraphA[N1][N2] = 0;
//                 tgraphA[N2][N1] = 0;
//                 signA = 1;
//             }

//             // Compute shortest paths from vertex 1
//             vector<vector<int>>
//                 shortest_path1A;
//             vector<int> index1A;
//             for (q = 0; q < nA; q++)
//                 index1A.push_back(q);
//             index1A[0] = N1;
//             index1A[N1] = 0;
//             vector<vector<int>> temp_graph1A = reindex(tgraphA, index1A);
//             vector<vector<int>> p1A = dijkstra(temp_graph1A);
//             for (i = 0; i < p1A.size(); i++)
//             {
//                 vector<int> tpath1A;
//                 for (j = 0; j < p1A[i].size(); j++)
//                 {
//                     tpath1A.push_back(inv(index1A)[p1A[i][j]]);
//                 }
//                 shortest_path1A.push_back(tpath1A);
//             }

//             // Compute shortest paths from vertex 2
//             vector<vector<int>>
//                 shortest_path2A;
//             vector<int> index2A;
//             for (q = 0; q < nA; q++)
//                 index2A.push_back(q);
//             index2A[0] = N2;
//             index2A[N2] = 0;
//             vector<vector<int>> temp_graph2A = reindex(tgraphA, index2A);
//             vector<vector<int>> p2A = dijkstra(temp_graph2A);
//             for (i = 0; i < p2A.size(); i++)
//             {
//                 vector<int> tpath2A;
//                 for (j = 0; j < p2A[i].size(); j++)
//                 {
//                     tpath2A.push_back(inv(index2A)[p2A[i][j]]);
//                 }
//                 shortest_path2A.push_back(tpath2A);
//             }

//             // Compute distance between vertex 1 and vertex 2
//             int DA = 0;
//             for (i = 0; i < shortest_path1A.size(); i++)
//                 if (shortest_path1A[i][shortest_path1A[i].size() - 1] == N2)
//                 {
//                     DA = shortest_path1A[i].size() - 1;
//                     break;
//                 }

//             // Compute shortest paths between vertex1 and vertex 2
//             vector<vector<int>>
//                 shortest_path12A;
//             for (i = 0; i < shortest_path1A.size(); i++)
//                 for (j = 0; j < shortest_path2A.size(); j++)
//                 {
//                     if (shortest_path1A[i][shortest_path1A[i].size() - 1] == shortest_path2A[j][shortest_path2A[j].size() - 1])
//                     {
//                         vector<int> temppathA = shortest_path1A[i];
//                         for (k = shortest_path2A[j].size() - 2; k >= 0; k--)
//                             temppathA.push_back(shortest_path2A[j][k]);
//                         if (temppathA.size() - 1 == DA)
//                             shortest_path12A.push_back(temppathA);
//                     }
//                 }

//             // Pair graph for vertex 1 and vertex 2
//             bool checkA = false;
//             set<int> SA;
//             for (i = 0; i < shortest_path12A.size(); i++)
//             {
//                 if (shortest_path12A[i][0] == N1 && shortest_path12A[i][shortest_path12A[i].size() - 1] == N2)
//                 {
//                     checkA = true;
//                     for (j = 0; j < shortest_path12A[i].size(); j++)
//                         SA.insert(shortest_path12A[i][j]);
//                 }
//             }
//             vector<int> VA;
//             VA.push_back(N1);
//             VA.push_back(N2);
//             PA[VA] = SA;

//             // Distance between vertex 1 and vertex 2
//             if (checkA)
//                 dA[VA] = signA * DA;
//             else
//                 dA[VA] = signA;

//             // Count number of vertices in pair graph
//             ntA[VA] = SA.size();

//             // Count number of edges in pair graph
//             int countA = 0;
//             for (i = 0; i < nA; i++)
//                 for (j = i + 1; j < nA; j++)
//                 {
//                     bool findpairA = false;
//                     set<int>::iterator i1A, i2A;
//                     i1A = SA.find(i);
//                     i2A = SA.find(j);
//                     if (i1A != SA.end() && i2A != SA.end())
//                         findpairA = true;
//                     if (findpairA && graphA[i][j] != 0)
//                         countA++;
//                 }
//             eA[VA] = countA;
//         }

//     // Make frequency table (sign frequency vectors)
//     map<vector<int>, int>::iterator itA;
//     map<vector<int>, vector<int>> frequencyA;
//     vector<int> dummyA;
//     for (i = 0; i < nA; i++)
//         dummyA.push_back(0);
//     for (i = 0; i < nA; i++)
//     {
//         for (j = 0; j < nA; j++)
//         {
//             vector<int> vecA;
//             vector<int> indA;
//             indA.push_back(i);
//             indA.push_back(j);
//             itA = dA.find(indA);
//             vecA.push_back(itA->second);
//             itA = ntA.find(indA);
//             vecA.push_back(itA->second);
//             itA = eA.find(indA);
//             vecA.push_back(itA->second);
//             frequencyA[vecA] = dummyA;
//         }
//     }

//     map<vector<int>, vector<int>>::iterator ittA = frequencyA.begin();
//     while (ittA != frequencyA.end())
//     {
//         for (i = 0; i < nA; i++)
//         {
//             int fA = 0;
//             for (j = 0; j < nA; j++)
//             {
//                 vector<int> vecA;
//                 vector<int> indA;
//                 indA.push_back(i);
//                 indA.push_back(j);
//                 itA = dA.find(indA);
//                 vecA.push_back(itA->second);
//                 itA = ntA.find(indA);
//                 vecA.push_back(itA->second);
//                 itA = eA.find(indA);
//                 vecA.push_back(itA->second);
//                 if (vecA == ittA->first)
//                     fA++;
//             }
//             ittA->second[i] = fA;
//         }
//         ittA++;
//     }

//     // Transpose and sort (canonical form of sign matrix)
//     vector<vector<int>>
//         vssA;
//     ittA = frequencyA.begin();
//     while (ittA != frequencyA.end())
//     {
//         vector<int> vsA = ittA->second;
//         vssA.push_back(vsA);
//         ittA++;
//     }
//     vector<vector<int>> tvssA;
//     vector<int> rowA;
//     for (i = 0; i < vssA.size(); i++)
//         rowA.push_back(0);
//     for (j = 0; j < vssA[0].size(); j++)
//         tvssA.push_back(rowA);
//     for (i = 0; i < vssA.size(); i++)
//         for (j = 0; j < vssA[0].size(); j++)
//             tvssA[j][i] = vssA[i][j];
//     for (i = 0; i < tvssA.size(); i++)
//         tvssA[i].push_back(i);
//     sort(tvssA.begin(), tvssA.end());

//     // Determine equivalence classes k
//     vector<int>
//         classA;
//     vector<int> clA, dlA;
//     int cA = 0;
//     int icA = 0;
//     dlA = tvssA[icA];
//     dlA.pop_back();
//     while (icA < nA)
//     {
//         clA = tvssA[icA];
//         clA.pop_back();
//         if (clA == dlA)
//             classA.push_back(cA);
//         else
//         {
//             dlA = clA;
//             cA++;
//             classA.push_back(cA);
//         }
//         icA++;
//     }

//     // Final Vertices
//     vector<int>
//         vertexA;
//     for (i = 0; i < nA; i++)
//         vertexA.push_back(tvssA[i][tvssA[i].size() - 1]);

//     // Final Sign Matrix
//     map<vector<int>, vector<int>>
//         signmatrixA;
//     for (i = 0; i < nA; i++)
//         for (j = 0; j < nA; j++)
//         {
//             vector<int> indA;
//             indA.push_back(vertexA[i]);
//             indA.push_back(vertexA[j]);
//             vector<int> siA;
//             itA = dA.find(indA);
//             siA.push_back(itA->second);
//             itA = ntA.find(indA);
//             siA.push_back(itA->second);
//             itA = eA.find(indA);
//             siA.push_back(itA->second);
//             signmatrixA[indA] = siA;
//         }

//     // Process Graph B
//     cout
//         << "Computing the Sign Matrix of Graph B..." << endl;
//     int nB;
//     infileB >> nB;

//     // Read adjacency matrix of graph B from graphB.txt
//     vector<vector<int>>
//         rgraphB;
//     int valB;
//     for (i = 0; i < nB; i++)
//     {
//         vector<int> rowB;
//         for (j = 0; j < nB; j++)
//         {
//             infileB >> valB;
//             rowB.push_back(valB);
//         }
//         rgraphB.push_back(rowB);
//     }

//     // End reading

//     // Initial sorting
//     vector<vector<int>>
//         distanceB;
//     vector<vector<int>> neighborB;
//     for (i = 0; i < rgraphB.size(); i++)
//     {
//         vector<int> rowdistanceB, rowneighborB;
//         for (j = 0; j < rgraphB[i].size(); j++)
//         {
//             if (rgraphB[i][j] == 1)
//                 rowneighborB.push_back(j);
//             vector<int> indexdB;
//             for (k = 0; k < rgraphB.size(); k++)
//                 indexdB.push_back(k);
//             indexdB[0] = i;
//             indexdB[i] = 0;
//             vector<vector<int>> dgraphB = reindex(rgraphB, indexdB);
//             vector<vector<int>> dpathB = dijkstra(dgraphB);
//             int ddB = -1;
//             for (k = 0; k < dpathB.size(); k++)
//                 if (inv(indexdB)[dpathB[k][dpathB[k].size() - 1]] == j)
//                 {
//                     ddB = dpathB[k].size() - 1;
//                     break;
//                 }
//             rowdistanceB.push_back(ddB);
//         }
//         distanceB.push_back(rowdistanceB);
//         neighborB.push_back(rowneighborB);
//     }

//     vector<vector<int>>
//         sequenceB;
//     for (i = 0; i < rgraphB.size(); i++)
//     {
//         vector<int> mutualdistanceB;
//         for (j = 0; j < neighborB[i].size() - 1; j++)
//             for (k = j + 1; k < neighborB[i].size(); k++)
//                 mutualdistanceB.push_back(distanceB[neighborB[i][j]][neighborB[i][k]]);
//         sort(mutualdistanceB.begin(), mutualdistanceB.end());
//         vector<int> tmutualB;
//         for (k = 0; k < nB * nB - 1 - mutualdistanceB.size(); k++)
//             tmutualB.push_back(0);
//         for (k = 0; k < mutualdistanceB.size(); k++)
//             tmutualB.push_back(mutualdistanceB[k]);
//         sequenceB.push_back(tmutualB);
//     }

//     map<vector<int>, int>
//         sorterB;
//     for (i = 0; i < sequenceB.size(); i++)
//         sorterB[sequenceB[i]] = i;
//     vector<int> mainindexB;
//     map<vector<int>, int>::iterator sitB = sorterB.begin();
//     while (sitB != sorterB.end())
//     {
//         vector<int> vsitB = sitB->first;
//         for (i = 0; i < sequenceB.size(); i++)
//             if (sequenceB[i] == vsitB)
//                 mainindexB.push_back(i);
//         sitB++;
//     }

//     vector<vector<int>>
//         graphB = reindex(rgraphB, mainindexB);

//     // Compute degree sequence
//     vector<int>
//         degB;
//     int sumB;
//     for (i = 0; i < nB; i++)
//     {
//         sumB = 0;
//         for (j = 0; j < nB; j++)
//         {
//             if (graphB[i][j] == 1)
//                 sumB++;
//         }
//         degB.push_back(sumB);
//     }
//     vector<int> sorted_degB = degB;
//     sort(sorted_degB.begin(), sorted_degB.end());

//     // Pair graphs
//     map<vector<int>, set<int>>
//         PB;
//     map<vector<int>, int> dB;
//     map<vector<int>, int> ntB;
//     map<vector<int>, int> eB;

//     for (N1 = 0; N1 < graphB.size(); N1++)
//         for (N2 = 0; N2 < graphB[N1].size(); N2++)
//         {
//             int signB = -1;
//             vector<vector<int>> tgraphB = graphB;
//             if (graphB[N1][N2] != 0)
//             {
//                 tgraphB[N1][N2] = 0;
//                 tgraphB[N2][N1] = 0;
//                 signB = 1;
//             }

//             // Compute shortest paths from vertex 1
//             vector<vector<int>>
//                 shortest_path1B;
//             vector<int> index1B;
//             for (q = 0; q < nB; q++)
//                 index1B.push_back(q);
//             index1B[0] = N1;
//             index1B[N1] = 0;
//             vector<vector<int>> temp_graph1B = reindex(tgraphB, index1B);
//             vector<vector<int>> p1B = dijkstra(temp_graph1B);
//             for (i = 0; i < p1B.size(); i++)
//             {
//                 vector<int> tpath1B;
//                 for (j = 0; j < p1B[i].size(); j++)
//                 {
//                     tpath1B.push_back(inv(index1B)[p1B[i][j]]);
//                 }
//                 shortest_path1B.push_back(tpath1B);
//             }

//             // Compute shortest paths from vertex 2
//             vector<vector<int>>
//                 shortest_path2B;
//             vector<int> index2B;
//             for (q = 0; q < nB; q++)
//                 index2B.push_back(q);
//             index2B[0] = N2;
//             index2B[N2] = 0;
//             vector<vector<int>> temp_graph2B = reindex(tgraphB, index2B);
//             vector<vector<int>> p2B = dijkstra(temp_graph2B);
//             for (i = 0; i < p2B.size(); i++)
//             {
//                 vector<int> tpath2B;
//                 for (j = 0; j < p2B[i].size(); j++)
//                 {
//                     tpath2B.push_back(inv(index2B)[p2B[i][j]]);
//                 }
//                 shortest_path2B.push_back(tpath2B);
//             }

//             // Compute distance between vertex 1 and vertex 2
//             int DB = 0;
//             for (i = 0; i < shortest_path1B.size(); i++)
//                 if (shortest_path1B[i][shortest_path1B[i].size() - 1] == N2)
//                 {
//                     DB = shortest_path1B[i].size() - 1;
//                     break;
//                 }

//             // Compute shortest paths between vertex 1 and vertex 2
//             vector<vector<int>>
//                 shortest_path12B;
//             for (i = 0; i < shortest_path1B.size(); i++)
//                 for (j = 0; j < shortest_path2B.size(); j++)
//                 {
//                     if (shortest_path1B[i][shortest_path1B[i].size() - 1] == shortest_path2B[j][shortest_path2B[j].size() - 1])
//                     {
//                         vector<int> temppathB = shortest_path1B[i];
//                         for (k = shortest_path2B[j].size() - 2; k >= 0; k--)
//                             temppathB.push_back(shortest_path2B[j][k]);
//                         if (temppathB.size() - 1 == DB)
//                             shortest_path12B.push_back(temppathB);
//                     }
//                 }

//             // Pair graph for vertex 1 and vertex 2
//             bool checkB = false;
//             set<int> SB;
//             for (i = 0; i < shortest_path12B.size(); i++)
//             {
//                 if (shortest_path12B[i][0] == N1 && shortest_path12B[i][shortest_path12B[i].size() - 1] == N2)
//                 {
//                     checkB = true;
//                     for (j = 0; j < shortest_path12B[i].size(); j++)
//                         SB.insert(shortest_path12B[i][j]);
//                 }
//             }
//             vector<int> VB;
//             VB.push_back(N1);
//             VB.push_back(N2);
//             PB[VB] = SB;

//             // Distance between vertex 1 and vertex 2
//             if (checkB)
//                 dB[VB] = signB * DB;
//             else
//                 dB[VB] = signB;

//             // Count number of vertices in pair graph
//             ntB[VB] = SB.size();

//             // Count number of edges in pair graph
//             int countB = 0;
//             for (i = 0; i < nB; i++)
//                 for (j = i + 1; j < nB; j++)
//                 {
//                     bool findpairB = false;
//                     set<int>::iterator i1B, i2B;
//                     i1B = SB.find(i);
//                     i2B = SB.find(j);
//                     if (i1B != SB.end() && i2B != SB.end())
//                         findpairB = true;
//                     if (findpairB && graphB[i][j] != 0)
//                         countB++;
//                 }
//             eB[VB] = countB;
//         }

//     // Make frequency table (sign frequency vectors)
//     map<vector<int>, int>::iterator itB;
//     map<vector<int>, vector<int>> frequencyB;
//     vector<int> dummyB;
//     for (i = 0; i < nB; i++)
//         dummyB.push_back(0);
//     for (i = 0; i < nB; i++)
//     {
//         for (j = 0; j < nB; j++)
//         {
//             vector<int> vecB;
//             vector<int> indB;
//             indB.push_back(i);
//             indB.push_back(j);
//             itB = dB.find(indB);
//             vecB.push_back(itB->second);
//             itB = ntB.find(indB);
//             vecB.push_back(itB->second);
//             itB = eB.find(indB);
//             vecB.push_back(itB->second);
//             frequencyB[vecB] = dummyB;
//         }
//     }

//     map<vector<int>, vector<int>>::iterator ittB = frequencyB.begin();
//     while (ittB != frequencyB.end())
//     {
//         for (i = 0; i < nB; i++)
//         {
//             int fB = 0;
//             for (j = 0; j < nB; j++)
//             {
//                 vector<int> vecB;
//                 vector<int> indB;
//                 indB.push_back(i);
//                 indB.push_back(j);
//                 itB = dB.find(indB);
//                 vecB.push_back(itB->second);
//                 itB = ntB.find(indB);
//                 vecB.push_back(itB->second);
//                 itB = eB.find(indB);
//                 vecB.push_back(itB->second);
//                 if (vecB == ittB->first)
//                     fB++;
//             }
//             ittB->second[i] = fB;
//         }
//         ittB++;
//     }

//     // Transpose and sort (canonical form of sign matrix)
//     vector<vector<int>>
//         vssB;
//     ittB = frequencyB.begin();
//     while (ittB != frequencyB.end())
//     {
//         vector<int> vsB = ittB->second;
//         vssB.push_back(vsB);
//         ittB++;
//     }
//     vector<vector<int>> tvssB;
//     vector<int> rowB;
//     for (i = 0; i < vssB.size(); i++)
//         rowB.push_back(0);
//     for (j = 0; j < vssB[0].size(); j++)
//         tvssB.push_back(rowB);
//     for (i = 0; i < vssB.size(); i++)
//         for (j = 0; j < vssB[0].size(); j++)
//             tvssB[j][i] = vssB[i][j];
//     for (i = 0; i < tvssB.size(); i++)
//         tvssB[i].push_back(i);
//     sort(tvssB.begin(), tvssB.end());

//     // Determine equivalence classes k
//     vector<int>
//         classB;
//     vector<int> clB, dlB;
//     int cB = 0;
//     int icB = 0;
//     dlB = tvssB[icB];
//     dlB.pop_back();
//     while (icB < nB)
//     {
//         clB = tvssB[icB];
//         clB.pop_back();
//         if (clB == dlB)
//             classB.push_back(cB);
//         else
//         {
//             dlB = clB;
//             cB++;
//             classB.push_back(cB);
//         }
//         icB++;
//     }

//     // Final Vertices

//     vector<int>
//         vertexB;
//     for (i = 0; i < nB; i++)
//         vertexB.push_back(tvssB[i][tvssB[i].size() - 1]);

//     // Final Sign Matrix
//     map<vector<int>, vector<int>>
//         signmatrixB;
//     for (i = 0; i < nB; i++)
//         for (j = 0; j < nB; j++)
//         {
//             vector<int> indB;
//             indB.push_back(vertexB[i]);
//             indB.push_back(vertexB[j]);
//             vector<int> siB;
//             itB = dB.find(indB);
//             siB.push_back(itB->second);
//             itB = ntB.find(indB);
//             siB.push_back(itB->second);
//             itB = eB.find(indB);
//             siB.push_back(itB->second);
//             signmatrixB[indB] = siB;
//         }

//     // Isomorphism Index
//     vector<int>
//         fixisoB, isoB;
//     for (i = 0; i < nB; i++)
//         fixisoB.push_back(i);
//     isoB = fixisoB;
//     bool isomorphic = false;
//     bool possibly_isomorphic = false;
//     if (sorted_degA == sorted_degB)
//     {
//         vector<vector<int>> checksignA;
//         ittA = frequencyA.begin();
//         while (ittA != frequencyA.end())
//         {
//             vector<int> checksignrowA;
//             vector<int> wsA = ittA->first;
//             for (i = 0; i < wsA.size(); i++)
//                 checksignrowA.push_back(wsA[i]);
//             vector<int> vsA = ittA->second;
//             for (i = 0; i < vsA.size(); i++)
//                 checksignrowA.push_back(vsA[vertexA[i]]);
//             checksignA.push_back(checksignrowA);
//             ittA++;
//         }
//         vector<vector<int>> checksignB;
//         ittB = frequencyB.begin();
//         while (ittB != frequencyB.end())
//         {
//             vector<int> checksignrowB;
//             vector<int> wsB = ittB->first;
//             for (i = 0; i < wsB.size(); i++)
//                 checksignrowB.push_back(wsB[i]);
//             vector<int> vsB = ittB->second;
//             for (i = 0; i < vsB.size(); i++)
//                 checksignrowB.push_back(vsB[vertexB[i]]);
//             checksignB.push_back(checksignrowB);
//             ittB++;
//         }
//         if (checksignA == checksignB)
//         {
//             possibly_isomorphic = true;
//         }
//     }

//     if (possibly_isomorphic)
//     {

//         // Find isomorphism
//         for (int J = 0; J < nB; J++)
//         {
//             if (isomorphic)
//                 break;
//             isoB = fixisoB;
//             isoB[0] = fixisoB[J];
//             isoB[J] = fixisoB[0];
//             for (int I = 0; I < nB * nB; I++)
//             {
//                 vector<int> oldisoB = isoB;
//                 isoB = transform(signmatrixA, signmatrixB, vertexA, vertexB, isoB);

//                 bool quit = false,
//                      mismatch = false;
//                 for (int ii = 0; ii < nB; ii++)
//                 {
//                     if (quit)
//                         break;
//                     for (int jj = ii + 1; jj < nB; jj++)
//                     {
//                         vector<int> tindA, ta;
//                         tindA.push_back(vertexA[ii]);
//                         tindA.push_back(vertexA[jj]);
//                         vector<int> tindB, tb;
//                         tindB.push_back(vertexB[isoB[ii]]);
//                         tindB.push_back(vertexB[isoB[jj]]);
//                         ittA = signmatrixA.find(tindA);
//                         ta = ittA->second;

//                         ittB = signmatrixB.find(tindB);
//                         tb = ittB->second;
//                         if (ta != tb)
//                         {
//                             mismatch = true;
//                             quit = true;
//                             break;
//                         }
//                     }
//                 }
//                 if (isoB == oldisoB)
//                 {
//                     if (!mismatch)
//                         isomorphic = true;
//                     break;
//                 }
//             }
//         }
//     }

//     if (!possibly_isomorphic)
//     {
//         outfile << "Graph A and Graph B cannot be isomorphic because "
//                 << "they have different sign frequency vectors in lexicographic order."
//                 << endl;
//         cout << "Graph A and Graph B cannot be isomorphic because "
//              << "they have different sign frequency vectors in lexicographic order."
//              << endl;
//     }
//     if (possibly_isomorphic && !isomorphic)
//     {
//         outfile << "Graph A and Graph B have the same sign frequency vectors "
//                 << "in lexicographic order but cannot be isomorphic." << endl;
//         cout << "Graph A and Graph B have the same sign frequency vectors "
//              << "in lexicographic order but cannot be isomorphic." << endl;
//     }
//     if (possibly_isomorphic && isomorphic)
//     {
//         outfile << "Graph A and Graph B are isomorphic." << endl
//                 << endl;
//         cout << "Graph A and Graph B are isomorphic." << endl;
//         outfile << "Isomorphism f:V(A)->V(B);" << endl
//                 << endl;
//         for (int i = 0; i < nA; i++)
//         {
//             outfile << "f(" << inv(mainindexA)[vertexA[i]] + 1 << ")\t=\t";
//             outfile << inv(mainindexB)[vertexB[isoB[i]]] + 1 << endl;
//         }
//     }

//     cout
//         << "See result.txt for details." << endl;
//     return 0;
// }

// // Functions

// vector<vector<int>>
// dijkstra(vector<vector<int>> graph)
// {
//     vector<vector<int>> table;
//     int i, j, k, n = graph.size();

//     // Initialize Table
//     const int infinity = n;
//     vector<bool> known;
//     for (i = 0; i < n; i++)
//         known.push_back(false);
//     vector<int> d;
//     d.push_back(0);
//     for (i = 1; i < n; i++)
//         d.push_back(infinity);
//     vector<int> p;
//     for (i = 0; i < n; i++)
//         p.push_back(-1);

//     // End initialization

//     // Iteration
//     for (k = 0; k < n; k++)
//     {

//         // Find min of d for unknown vertices
//         int min = 0;
//         while (known[min] == true)
//             min++;
//         for (i = 0; i < n; i++)
//             if (known[i] == false && d[i] < d[min])
//                 min = i;

//         // End find
//         //  Update Table
//         known[min] = true;
//         for (j = 0; j < n; j++)
//         {
//             if (graph[min][j] != 0 && d[j] > d[min] + graph[min][j] && known[j] == false)
//             {
//                 d[j] = d[min] + graph[min][j];
//                 p[j] = min;
//             }
//         }

//         // End update
//     }
//     table.push_back(d);
//     table.push_back(p);
//     vector<vector<int>> path;
//     for (i = 0; i < n; i++)
//     {
//         vector<int> temp_path;
//         vector<int> temp;
//         k = i;
//         while (k != -1)
//         {
//             temp.push_back(k);
//             k = table[1][k];
//         }
//         temp_path.push_back(temp[temp.size() - 1]);
//         for (j = temp.size() - 2; j >= 0; j--)
//         {
//             temp_path.push_back(temp[j]);
//         }
//         path.push_back(temp_path);
//     }
//     return path;
// }

// vector<vector<int>>
// reindex(vector<vector<int>> graph, vector<int> index)
// {
//     vector<vector<int>> temp_graph = graph;
//     for (int i = 0; i < graph.size(); i++)
//         for (int j = 0; j < graph[i].size(); j++)
//             temp_graph[index[i]][index[j]] = graph[i][j];
//     return temp_graph;
// }

// vector<int>
// inv(vector<int> index)
// {
//     vector<int> inverse = index;
//     for (int i = 0; i < index.size(); i++)
//         inverse[index[i]] = i;
//     return inverse;
// }

// vector<int>
// transform(map<vector<int>, vector<int>> signmatrixA,
//           map<vector<int>, vector<int>> signmatrixB,
//           vector<int> vertexA, vector<int> vertexB,
//           vector<int> isoB)
// {
//     vector<int> iso = isoB;
//     map<vector<int>, vector<int>>::iterator it;
//     int k, n = iso.size();
//     bool found = false;
//     bool check = true;
//     for (int i = 0; i < n; i++)
//     {
//         if (found)
//             break;
//         for (int j = i + 1; j < n; j++)
//         {
//             vector<int> indA, a;
//             indA.push_back(vertexA[i]);
//             indA.push_back(vertexA[j]);
//             vector<int> indB, b;

//             indB.push_back(vertexB[isoB[i]]);
//             indB.push_back(vertexB[isoB[j]]);
//             it = signmatrixA.find(indA);
//             a = it->second;
//             it = signmatrixB.find(indB);
//             b = it->second;
//             if (a != b)
//             {
//                 k = j;
//                 vector<int> temp_ind = indB;
//                 while (k < n - 1 && check == true)
//                 {
//                     k++;
//                     temp_ind[1] = vertexB[isoB[k]];
//                     it = signmatrixB.find(temp_ind);
//                     b = it->second;

//                     // check
//                     if (a == b)
//                     {
//                         vector<int> temp_iso = isoB;
//                         temp_iso[j] = isoB[k];
//                         temp_iso[k] = isoB[j];
//                         int ti = -1, tj = -1;
//                         bool quit = false;
//                         for (int ii = 0; ii < n; ii++)
//                         {
//                             if (quit)
//                                 break;
//                             for (int jj = ii + 1; jj < n; jj++)
//                             {
//                                 vector<int> tindA, ta;
//                                 tindA.push_back(vertexA[ii]);
//                                 tindA.push_back(vertexA[jj]);
//                                 vector<int> tindB, tb;
//                                 tindB.push_back(vertexB[temp_iso[ii]]);
//                                 tindB.push_back(vertexB[temp_iso[jj]]);
//                                 it = signmatrixA.find(tindA);
//                                 ta = it->second;
//                                 it = signmatrixB.find(tindB);
//                                 tb = it->second;
//                                 if (ta != tb)
//                                 {
//                                     ti = ii;
//                                     tj = jj;
//                                     quit = true;
//                                     break;
//                                 }
//                                 if (k == n - 1 && ti == -1)
//                                 {
//                                     check = false;
//                                     quit = true;
//                                     break;
//                                 }
//                             }
//                         }
//                         if (ti == -1 || ti > i || (ti == i && tj > j))
//                             check = false;
//                     }

//                     // end check
//                 }
//                 if (!check)
//                 {
//                     found = true;
//                     iso[j] = isoB[k];
//                     iso[k] = isoB[j];
//                     break;
//                 }
//                 if (check)
//                     return iso;
//             }
//         }
//     }
//     return iso;
// };