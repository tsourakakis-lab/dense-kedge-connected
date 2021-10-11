
import collections
import numpy as np

import cvxopt
import cvxpy as cp
from cvxopt import matrix
from cvxopt.blas import dot

import matplotlib.pyplot as plt
import networkx as nx

from itertools import *
from gurobipy import *


def density(G):
    return float(G.number_of_edges()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0

# Return the minimum degree, or 0 if there are no vertices
def min_deg(G):    
    return min( [ d for (_, d) in G.degree() ] or [0] )    

# Returns both the vertex and the degree. 
def min_deg_vertex(G):
    return min(G.degree(), key=lambda (v, d):d)

def graph_stats(G, p="", level=0, verbose=False):
    s = "   " * level
    print()
    print(p)
    avg_deg = 2.0 * len(G.edges()) / len(G.nodes())
    print(s, "n =", len(G.nodes()), "m =", len(G.edges()), "avg_deg =", avg_deg, "density =", avg_deg / 2)
    k = max(nx.core_number(G).values())
    print(s, "core number =", k)
    C = nx.k_core(G, k)
    print("size of core:", C.number_of_nodes(), C.number_of_edges())

    #print s, "edge connectivity =", nx.edge_connectivity(G)
    #print s, "vertex connectivity =", nx.node_connectivity(G)
    
    #print s, "max vertex conn = ", nx.all_pairs_edge_connectivity(G)
    if verbose: print(s, sorted(G.nodes()))


def preprocess_graph(G):    
    G = G.to_undirected()
    G = nx.convert_node_labels_to_integers(G)
    G.remove_edges_from(G.selfloop_edges())
    return G

def read_graph(f):
    return preprocess_graph(nx.read_edgelist(f, nodetype=int))

def graph_info(G, string, k=None):    
    graph_stats(G, string, 0)
    
    if k is None: k = max(nx.core_number(G).values())
        
    avg_deg, S = densest_subgraph(G)
    graph_stats(S, "densest:", 1)
    
    H = nx.k_core(G, k)
    graph_stats(H, "%i-core:" % k, 1)    
    return G, avg_deg, S, H


def densest_subgraph(G): # assumes G is undirected
    vertices = G.nodes()
    und_edges = G.edges()
    if not und_edges: return 0, []    

    model = Model()

    # Suppress output
    model.params.OutputFlag = 0

    # Add variables
    y = model.addVars(vertices, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="y")
    x = model.addVars(und_edges, lb=0, vtype=GRB.CONTINUOUS, name="x")
    model.update()

    # Size constraint
    model.addConstr(quicksum(y[i] for i in vertices) == 1)

    # Edge constraints
    for v, w in und_edges:
        model.addConstr(x[v, w] <= y[v])
        model.addConstr(x[v, w] <= y[w])

    # Set objective function (average degree)
    model.setObjective(quicksum(2 * x[v, w] for (v, w) in und_edges))
    model.modelSense = GRB.MAXIMIZE

    model.update()
    model.optimize()

    assert model.status == GRB.status.OPTIMAL
    sol = [ v for v in vertices if y[v].x > 0 ]
    induced = G.subgraph(sol)        
    d = 2.0 * induced.number_of_edges() / induced.number_of_nodes()
    assert d >= 2.0 * len(und_edges) / len(vertices)
    return d, induced    

import subprocess, time
mkecs_bin = '/home/tony/Python_Project/k-edge-connected-master/bin/mkecs'
edgelist_file_path = '/home/tony/Python_Project/k-edge-connected-master/temp.edgelist'
ts = str(time.time()).split('.')[0]

# Use the mkecs binary from https://github.com/iwiwi/k-edge-connected/ 
# to find k-connected components
def k_connected_components(G, k):
    # For k=1 use networkx. Note that biconnected components are not the same as k-connected components for k=2.
    if k == 1: return [ list(c) for c in nx.connected_components(G) ]
    
    global mkecs_bin, edgelist_file_path, ts    
    
#     pipein, pipeout = os.pipe()
    nx.write_edgelist(G, edgelist_file_path+ts, data=False)
#     edges = ''.join("%i %i\n" % e for e in G.edges())
#     os.write(pipeout, edgelist_file_path)
#     os.close(pipeout)

    with open(edgelist_file_path+ts, 'r') as f:
        result = subprocess.check_output([mkecs_bin, str(k)], stdin=f)
#     print(result)
#     result = subprocess.check_output([mkecs_bin, str(k)], stdin=pipein)
#     os.close(pipein)
    if result != '':
        if result[-1] == '\n': result = result[:-1]
        return map(lambda s:map(int, s.split()), result.split('\n'))        
    else:        
        return []    
    
    
def is_k_connected(G, k):
    comps = k_connected_components(G, k)
    return comps != [] and len(comps[0]) == G.number_of_nodes()


import time

# Returns k and the k-connected components of G for the largest k possible
def most_highly_connected_subgraphs(G, verbose=False):
    if verbose:
        graph_stats(G)    
        avg_deg, S = densest_subgraph(G)
        graph_stats(S, "densest:", 1)    
        start = time.time()
    
    first_bound = max(nx.core_number(G).values())    
    if verbose: print "first bound =", first_bound

    def try_k(G, k):
        components = k_connected_components(G, k)
        if components != []:
            if verbose: print k, "YES"
            if k > try_k.best[0]: try_k.best = k, components
        return components
    
    try_k.best = (0, [])
        
    a = first_bound
    while not try_k(G, a): a = int(a / 2)
        
    if verbose: assert try_k(G, a)

    b = a + 1
    while try_k(G, b):
        a, b = b, int(b * 1.2 + 1)
    while b - a != 1:
        m = (a + b) / 2
        if try_k(G, m):
            a = m
        else:
            b = m
    if verbose:        
        print("time taken =", time.time() - start)    
        print("max k-connected = ", a)
    return try_k.best

# Our 3.16-approximation for the G=H case
def dense_and_highly_connected(G, verbose=False):
    k, comps = most_highly_connected_subgraphs(G, verbose=verbose) 
    graphs = ( G.subgraph(c) for c in comps )
    sol = max(graphs, key=lambda S:density(S) )
    return k, density(sol), sol.nodes() 


global it, level

sys.setrecursionlimit(20000)

def k_conn_min_deg_aux(G, H, k, verbose=False, pre_min_deg=0):
    if verbose:
        global it, level
        it = it + 1
        level = level + 1
        print'  ' * level, "iteration =", it, "level =", level, "nodes =", len(G.nodes()), len(H.nodes())    
        print'  ' * level, "Computing %i-connected components..." % k
        
    if len(G.nodes)==0:
        return []
    cn = nx.core_number(H)
    deg = min(cn.values())
    if deg<=pre_min_deg:
        others = [ v for (v, i) in cn.iteritems() if i > pre_min_deg ]
        subgraphs = ([], k_conn_min_deg_aux(G.subgraph(others), H.subgraph(others), k, verbose=verbose, pre_min_deg=pre_min_deg))
    else:
        
        comps = k_connected_components(G, k)
        if verbose: print("done")

        if comps == []:                 
            # No k-connected component: no solution
            return []                      
        elif len(comps) > 1 or len(comps[0]) != G.number_of_nodes():                   
            # G is not k-connected
            subgraphs = ( k_conn_min_deg_aux(G.subgraph(c), H.subgraph(c), k, verbose=verbose, pre_min_deg=pre_min_deg) for c in comps )    
        else:                                   
            # G is k-connected
            # Remove the lowest core. This is better than just removing the lowest-degree vertex as it
            # avoids many useless computations of k-conected components
  
            others = [ v for (v, i) in cn.iteritems() if i > deg ]
            if verbose:
                print '  ' * level, 'connected component found', len(G.nodes()), min_deg(H), deg, len(others), max(cn.values())
            subgraphs = ( H.nodes(), k_conn_min_deg_aux(G.subgraph(others), H.subgraph(others), k, verbose=verbose, pre_min_deg=deg) )
        
    if verbose: level = level - 1
    
    # Return the best (highest min-degree) solution among those in subgraphs. Break ties by density.
    return max(subgraphs, key=lambda S: (min_deg(H.subgraph(S)), density(H.subgraph(S)) ) )               
        
def k_conn_min_deg(G, H, k, verbose=False):    
    if verbose:
        global it, level
        it, level = 0, 0
    S = k_conn_min_deg_aux(G, H.copy(), k, verbose=verbose)
    GS, HS, mindeg = G.subgraph(S), H.subgraph(S), min_deg(H.subgraph(S))
    # assert GS.nodes() == HS.nodes()
    if verbose: print "size(S) =", len(S)
    assert S == [] or is_k_connected(G.subgraph(S), k)
    return S, mindeg

def kcco(G, H, k):
    results = []
    comps = k_connected_components(G, 1)
    for comp in comps:
        S = H.subgraph(comp)
        core = nx.k_core(S, k)
        
        if (core.number_of_nodes()<=1):
            continue
        if nx.is_connected(G.subgraph(core.nodes())):
            results.append(list(core.nodes()))
        else:
            core_comps = k_connected_components(G.subgraph(core.nodes()), 1)
            for core_comp in core_comps:
                results += kcco(G.subgraph(core_comp), H.subgraph(core_comp), k)
    if len(results)==0:
        return []
    return [max(results,key=lambda x:len(x))]
