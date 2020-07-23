from strawberryfields.apps import data, sample, subgraph, plot
import plotly
import networkx as nx
import numpy as np

class GBS:
    def __init__(self, samples, min_pho = 16, max_pho = 30, subgraph_size = 8, max_count = 2000):
        self.samples = samples
        self.min_pho = min_pho
        self.max_pho = max_pho
        self.subgraph_size = subgraph_size
        self.max_count = max_count

    def graphDensity(self, samples, min_pho, max_pho, subgraph_size, max_count):
        dense = subgraph.search(samples, pl_graph, subgraph_size, min_pho, max_count=max_count)
        dense_freq = []
        for k in range(subgraph_size, min_pho+1):
            dense_freq.append([k,len(dense[k])])
        return dense, dense_freq

    def graphFreqScore(self, d_freqs, max_freq):
        x,y = [], []
        for i in range(len(d_freqs)):
            for j in range(len(d_freqs[i])):
                n,f = d_freqs[i][j][0],d_freqs[i][j][1]
                x.append(n*f)
            N = len(d_freq[i])
            y.append((1/max_freq)*(np.sum(x)/N))
            x = []
        min_y = np.min(y)
        y = [min_y/x for x in y]

        return y, y.index(max(y))
