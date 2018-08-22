""" Simple script to check the number of Nodes left
"""
import math

nodes = 10
routing_iters = 2

# list of percentage per sparse operation
sparse_list = [0.3, 0.3]

for _ in range(routing_iters):
    for p in sparse_list:
        nodes = math.ceil(nodes * (1-p))
        print(nodes)
print(f"final: {nodes}")