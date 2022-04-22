# Project:  CS 617
# Program:  Dijkstra's Algorithm
# Purpose:  Implement and test Dijkstra's single source shortest path algorithm
# Author:   Digya Acharya
# Source:   - T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein,
#             Introduction to Algorithms, Third Edition, The MIT Press, Cambridge MA, 2009
#           - M. Petty, Dijkstra's Algorithm, Lecture 22, 2022
#           - M. Petty, Heapsort and Priority Queues, Lecture 11, 2022
#           - M. Petty, Heaps, Lecture 10, 2022 
# Created:  2022-04-18
# Modified: 2022-04-19

import math

# Set output file
output_file = open('output_file.txt', 'w')

# Add edges to the graph and apply Dijkstra's algorithm 
class Graph():
    def __init__(self, vertices):
        self.vertices = vertices
        self.graph = {}

    def add_edges(self, edges):     # Add each new edge to the beginning of the list of nodes
        for (source, dest, weight) in edges:
            if self.graph.get(source) is None:
                self.graph[source] = []
                self.graph[source].insert(0, [dest, weight])
            else:
                self.graph[source].insert(0,[dest, weight])
        for (source, dest, weight) in edges:
            if dest not in self.graph:
                self.graph[dest] = []
 
    # Referenced from Dr. Petty's lecture 22 slides
    def dijkstra(self, s):  # Vertices have been added to the heap during initialization instead of passing Graph  
        W = self.vertices
        S = []
        pi = {}              
        heap, d = self.initialize_single_source(s, W)          
        heap.build_min_heap()      
        heap.heap_decrease_key(s, d[s])
        heap.size = W
        while heap.size > 0:
            u = heap.heap_extract_min()[0] # heap_extract_min() gives all vertices and it's distance from the 
            S.append(u)                    # extracted minimum vertex
            for (v, w) in self.graph[u]:   
                pi, d = self.relax(u, v, w, d, heap, pi)      
        return dict(sorted(pi.items())), d

    def initialize_single_source(self, s, W):
        d = []
        heap = MinHeap()        
        for v in range(W):
            d.append(math.inf)       
            heap.array.append(heap.new_node(v, d[v])) # Add each vertex
            heap.pos.append(v)                        # Each vertex number or position
        heap.pos[s] = s                               # Source vertex number or position
        d[s] = 0 
        return heap, d

    def relax(self, u, v, w, d, heap, pi):
        if (d[v] > w + d[u]):
            d[v] = d[u] + w
            pi[v] = u + 1                             # Updated for 0-based indexing
            heap.heap_decrease_key(v, d[v])
        return pi, d

# Create MinHeap Priority Queue
class MinHeap():
    def __init__(self):
        self.array = []
        self.size = 0
        self.pos = []

    # Functions from Dr. Petty's lecture 10 and 11 pseudocode
    def left(self, i):
        return 2*i + 1                               # Updated for 0-based indexing

    def right(self, i):
        return 2*i + 2                               # Updated for 0-based indexing

    def parent(self, i):
        return (i - 1) // 2                          # Updated for 0-based indexing

    def new_node(self, v, d):                        # Each node in a heap consists of vertex and distance from source
        return [v, d]

    def swap_nodes(self, x, y):                  
        temp = self.array[x]
        self.array[x] = self.array[y]
        self.array[y] = temp

    def min_heapify(self, i):
        smallest = i
        l = self.left(i)
        r = self.right(i)
        if (l < self.size and self.array[l][1] < self.array[smallest][1]): # Take the second element of array as it 
            smallest = l                                                   # consists of distance
        elif (r < self.size and self.array[r][1] < self.array[smallest][1]):
            smallest = r
        if smallest != i:
            self.pos[self.array[smallest][0]] = i    # Update position value and swap nodes
            self.pos[self.array[i][0]] = smallest
            self.swap_nodes(smallest, i)
            self.min_heapify(smallest)

    def build_min_heap(self):
        for i in range(int(math.floor(len(self.array)/2))-1, -1, -1): # Updated for 0-based indexing
            self.min_heapify(i)
 
    def heap_extract_min(self):         # Along with extracting the minimum of the heap, remove the position from                                 
        if self.size < 1:               # the position list as well
            return
        min = self.array[0]
        self.array[0] = self.array[self.size - 1]
        self.pos[self.array[0][0]] = 0
        self.pos[min[0]] = self.size - 1
        self.size -= 1
        self.min_heapify(0)
        return min
 
    def heap_decrease_key(self, x, key):    # Along with adding to the heap, add to the position list as well
        i = self.pos[x]                     # Get the index number of x
        self.array[i][1] = key
 
        while i > 0 and self.array[i][1] < self.array[self.parent(i)][1]:
            self.pos[self.array[i][0]] = self.parent(i)
            self.pos[self.array[self.parent(i)][0]] = i
            self.swap_nodes(i, self.parent(i))
            i = self.parent(i)
 
def find_shortest_path(prev_nodes, distances, start_node, vertices):   # Find the shortest path from source vertex to
    paths = []                                                         # all the other vertices
    weights = []
    for i in vertices:
        node = i
        shortest_path = []
        while node != start_node:
            if prev_nodes[node]:
                shortest_path.append (node)
                node = prev_nodes[node] - 1
        shortest_path.append (start_node)
        result = [vertices[key] for key in reversed(shortest_path)]
        paths.append(result)
        weights.append(distances[i])
    return paths, weights                           # Return shortest paths and weight of those paths

def create_graphs(): 
    # edges1 consists of edges of graph 1, and edges2 consists of edges of graph 2
    edges1 = [(0, 1, 1), (0, 2, 5), (1, 2, 6), (1, 3, 3), (2, 4, 2), (1, 4, 2), (3, 4,3), (3,5, 2), (4,5,2)]
    edges2 = [(1, 0, 8), (2, 1, 2), (2,3,9), (7,3, 10), (11,7,2), (10,11,8), (10,9,1), (8,9,10), (4,8,2), (4,0,10), 
    (4,9,8), (5,8,11), (1,4,2), (0,5,1), (5,10,1), (9,6,2), (1,6,9), (2,5,10), (6,11,1), (7,10,9), (7,2,1), (6,3,9), 
    (9, 5, 10), (1, 5, 8), (6, 10, 8), (2, 6, 8)]

    graph1 = Graph(6)
    graph1.add_edges(edges1)
    graph2 = Graph(12)
    graph2.add_edges(edges2)
    return graph1, graph2

def write_to_out_file(graph_no, prev, source_vertex, distances, paths, weights, vertices): # Writes to the output file
    global output_file
    prev_str = str(prev).replace("[", "").replace("]", "").replace("'", "").replace(",", "")
    dist_str = str(distances).replace("[", "").replace("]", "").replace(",", "")
    output_file.write("graph= "+ str(graph_no)+ '\n')
    output_file.write("result p= " + prev_str + '\n')
    output_file.write("result d= " + dist_str + '\n')
    for (i, (key, value)) in enumerate(vertices.items()):
        path_str = str(paths[i]).replace("[", "").replace("]", "").replace("'", "").replace(",", "")
        output_file.write("from= " + str(vertices[source_vertex]) + " to= " + value + " path= " + path_str  
        + " weight= " + str(weights[i]) + '\n')
    output_file.write('\n')

if __name__ == "__main__":
    # vertices1 are vertices of graph 1 and vertices2 are vertices of graph 2
    vertices1 = {0:'x', 1:'a', 2:'b', 3:'c', 4:'d', 5:'z'}
    vertices2 = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j', 10:'k', 11:'l'}

    graph1, graph2 = create_graphs()

    p1, d1 = graph1.dijkstra(0)
    p1 = list(p1.values())
    p1.insert(0, 'NA')                            # Insert NA for the previous node of source vertex
    paths, weights = find_shortest_path(p1, d1, 0, vertices1)
    write_to_out_file(1, p1, 0, d1, paths, weights, vertices1)

    p2, d2 = graph2.dijkstra(0)
    p2 = list(p2.values())
    p2.insert(0, 'NA')
    paths, weights = find_shortest_path(p2, d2, 0, vertices2)
    write_to_out_file(2, p2, 0, d2, paths, weights, vertices2)

    p3, d3 = graph2.dijkstra(7)
    p3 = list(p3.values())
    p3.insert(7, 'NA')
    paths, weights = find_shortest_path(p3, d3, 7, vertices2)
    write_to_out_file(3, p3, 7, d3, paths, weights, vertices2)

    
    
