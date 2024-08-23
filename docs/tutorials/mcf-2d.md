# 2D Phase unwrapping using Minimum Cost Flow (MCF)

2D unwrappers can be easily built by combining the MCF solvers and planar graph interfaces available in `spurt`. In the examples below, we use the OR Tools solver but the interface classes can be used to add support for other MCF solvers as well.


## Regular grid 2D MCF

A regular 2D grid unwrapper can be implemented as follows using `spurt`

``` py
# igram is a 2D array of np.complex64
g = spurt.graph.Reg2DGraph(igram.shape)
solver = spurt.mcf.ORMCFSolver(g)

# Assume unit cost
cost = np.ones(solver.edges.shape[0], dtype=int)
unw, _ = solver.unwrap_one(igram.flatten(), cost)
unw = unw.reshape(igram.shape)
```

## Irregular grid 2D MCF

A sparse 2D grid unwrapper can be implemented as follows using `spurt`

``` py

# igram is a 1D array of type np.complex64 and length npts
# xy is a npts x 2 array with coordinates of pixels

graph = spurt.graph.DelaunayGraph(xy)
solver = spurt.mcf.ORMCFSolver(graph)

# Use unit cost
cost = np.ones(solver.edges.shape[0], dtype=int)
unw, _ = solver.unwrap_one(igram, cost)
```
