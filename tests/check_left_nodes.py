""" Simple script to check the number of Nodes left
"""
import math

nodes = 10
routing_iters = 3

# list of percentage per sparse operation
sparse_list = [(0.3, 0.3)]

for i in range(routing_iters - 1):
    print(f"Routing iter {i}")
    for j, sparse in enumerate(sparse_list):
        if len(sparse_list) > 1:
            print(f"Sparse operation: {j}")
        p = sparse[i]
        nodes = math.ceil(nodes * (1-p))
        print(nodes)
print(f"final: {nodes}")