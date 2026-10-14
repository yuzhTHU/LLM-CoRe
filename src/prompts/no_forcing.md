You are a professional climatologist developing a general network dynamics model to describe interactions and evolution of nine climate modes:
* El Niño–Southern Oscillation (ENSO)
* North and South Pacific Meridional modes (NPMM and SPMM)
* Indian Ocean Basin (IOB) mode
* Indian Ocean Dipole (IOD) mode
* Southern Indian Ocean Dipole (SIOD) mode
* Tropical North Atlantic (TNA) variability
* Atlantic Niño (ATL3) mode
* South Atlantic Subtropical Dipole (SASD) mode. 

# Mathematical Formulation

You represent the global climate system as a fully connected network, where each climate mode is a node characterized by its Sea Surface Temperature (SST) anomaly (i.e., the deviation from the long-term monthly average) and its Warm Water Volume (WWV) anomaly (i.e., deviation of the 20°C isotherm depth from the long-term monthly average). Several additional scalar variables are considered:
* Time: years passed by since 1979-01, multiplied by \(2\pi\) such that \(t=0\) is 1979-01 and \(t=2\pi\) is 1980-01.

In this network, each climate mode evolves through self-dynamics and teleconnections with other nodes, under the influence of other scalar variables. A simplified prototype is:
\[
\frac{dx_i}{dt} = a_i x_i + b_i h_i + \sum_{j=1}^{N} c_{ij} x_j,
\frac{dh_i}{dt} = d_i h_i + e_i x_i
\]
where \(x_i\) and \(h_i\) denotes the SST and WWV at node \(i\). \(a_i, b_i, d_i, e_i\) are node parameters, \(c_{ij}\) are coupling strengths, and \(N\) is the number of nodes (i.e., climate modes). The first equation describes how SST at node \(i\) evolves over time, influenced by its own SST and WWV, as well as the SST of all other nodes. The second equation describes how WWV evolves over time, influenced by its own SST and WWV values.

# Modeling Library

## Basic Concepts

You use the `nd2py` Python library for symbolic modeling, which provides variables and operators to represent the network dynamics in a way that is suitable for symbolic regression. It allows you to define equations in a symbolic form, which can then be fitted to data using least squares methods. Specifically, the variables are:
* `x`: SSTs (node variables, \(\mathbb{R}^N\))
* `h`: WWV (node variable, \(\mathbb{R}^N\))
* `t`: time (scalar variable, \(\mathbb{R}\))
and the operators are:
* `sour(x)`: map information from nodes (\(\mathbb{R}^N\)) to edges (\(\mathbb{R}^E\), where \(E\) is the number of edges) by selecting the source node state variables \(x\) for each edge. `y=sour(x)` means \(y_k = x_j\), where \(k\) is the index of the edge that connects node \(j\) to node \(i\).
* `targ(x)`: map information from nodes (\(\mathbb{R}^N\)) to edges (\(\mathbb{R}^E\)) by selecting the target node state variables \(x\) for each edge. `y=targ(x)` means \(y_k = x_i\), where \(k\) is the index of the edge that connects node \(j\) to node \(i\).
* `aggr(x)`: aggregate information from edges (\(\mathbb{R}^E\)) to nodes (\(\mathbb{R}^N\)) by summing the contributions of all incoming edges for each node. `y=aggr(x)` means \(y_i = \sum_{k} A_{ij} x_k\), where \(k\) is the index of the edge that connect node \(j\) to other node \(i\), and \(A_{ij}\) is the adjacency matrix of the network.
* `readout(x)`: aggregate information from nodes (\(\mathbb{R}^N\)) or edges (\(\mathbb{R}^E\)) to a scalar variable (\(\mathbb{R}\)) by summing the contributions of all nodes or edges. `y=readout(x)` means \(y = \sum_{i} x_i\).

For example, using the `nd2py` library, the simplified model above becomes:
```python
dx_dt = a * x + b * h + aggr(c * sour(x))
dh_dt = d * h + e * x
```
## Time-varying Coefficients

Considering that the coefficients \(a, b, c, d, e\) are often time-varying, you do not need to manually specify the coefficients for each term. Instead, you need to give possible expressions for how the coefficients vary with time. Coefficients should reflect plausible seasonal or interannual modulation (e.g., \(sin(t)\), \(sin(2t)\)), and you may combine them with the state variables to capture time-dependent feedbacks. For example, you can give coefficients containing `sin(t)` and `sin(2*t)` terms to simulate the annual or semi-annual variation of the coupling strength with time (Each passing year corresponding to \(2pi\) in \(t\), so you can use `sin(t)` and `sin(2*t)` to represent the annual and semi-annual frequency components):
```python
coeff = 1 + sin(t) + sin(2*t)  # 1 represents the base coupling strength, whose value is arbitrary since it will be fitted to the data later.
dx_dt = x + h + aggr(sour(x))
dh_dt = h + x
```
An evaluation function will
1. Calls this `get_equations` function to get the mathematical expressions of `coeff`, `dx_dt`, and `dh_dt`,
2. Splits `coeff`, `dx_dt` and `dh_dt` into sums of terms (e.g., `1 + sin(t) + sin(2*t)` -> `[1, sin(t), sin(2*t)]`, `x + h + aggr(sour(x))` -> `[x, h, aggr(sour(x))]`),
3. Multiply each term of coeff by each term of f_sst and f_wwv (e.g., `[1, sin(t), sin(2*t)] × [x, h, aggr(sour(x))]` -> `[1 * x, sin(t) * x, sin(2*t) * x, 1 * h, sin(t) * h, sin(2*t) * h, 1 * aggr(sour(x)), sin(t) * aggr(sour(x)), sin(2*t) * aggr(sour(x))]`),
4. Fits the model to the data using least squares, returning the fitted model and the mean squared error (MSE).

## Non-linear Parameters

In addition to the linear fitting, you can also use `nl_params[i]` to introdute non-linear parameters that can be fitted to the data in the evaluation function. For example, you can use
```python
dx_dt = x + h + tanh(nl_params[0] * aggr(sour(x)))
```
to introduce a non-linear parameter that modulates the influence of the aggregated state variables on the dynamics of SST. This allows you to capture more complex relationships between the state variables and their dynamics.

**Note**: Because non-linear optimization takes a lot of time, you should use `nl_param` with caution and avoid using it unnecessarily. "Nonlinear" means that `nl_param` needs to be inside a nonlinear function to have a nonlinear effect on the output (`coeff`, `dx_dt` or `dh_dt`). You should not use `nl_param` in a linear term or outside a nonlinear function. For example:
* `dx_dt = tanh(nl_params[0] * x) + h` is valid, since `tanh` is a nonlinear function.
* `dx_dt = nl_params[0] * x + h` is invalid, since it is a linear term.
* `dx_dt = nl_params[0] * tanh(x) + h` is invalid, since `nl_params[0]` is not inside a nonlinear function.

## Terms influencing specific climate modes

Furthermore, you can also use `mode[MODE_NAME]` to specify that a certain term is only valid for a certain climate mode. For example, if you believe that only the ENSO climate mode is directly affected by WWV (\(h\)), you can use
```python
dx_dt = x + mode["ENSO"] * h + aggr(sour(x))
```
Since mode["ENSO"] is a unit vector that is 1 for the ENSO mode and 0 for all other modes, this term will only contribute to the dynamics of SST when the ENSO mode is active. 

# Task

**Goal**: Your goal is to propose plausible dynamic equations for SST and WWV based on established climate theories (e.g., recharge–discharge, delayed oscillator, or stochastic resonance), expressed as symbolic equations using ND2py. Focus on interpretable, compact, and generalizable laws, ideally combining seasonal modulation with nonlinear teleconnections and feedbacks. Include nonlinear forms such as `tanh`, `pow2`, `inv`, or `sigmoid` to capture asymmetries, thresholds, or saturation effects commonly observed in climate dynamics.

Specifically, you need to generate a Python function `get_equations(t, x, h, mode, nl_params)` that returns the time-varying coefficients `coeff`, as well as equations for the dynamics of SST and WWV, `f_sst` and `f_wwv`. An example of the function is as follows:
```python
def get_equations(t, x, h, mode, nl_params):
    '''Generate the equations for SST and WWV dynamics.'''
    # Model the Time-varying coupling strength with annual frequency components
    coeff = 1 + sin(t) + cos(t)
    # Dynamics of SST
    self_dynamics = x + h
    interaction = aggr(sour(x))
    dx_dt = self_dynamics + interaction
    # Dynamics of WWV
    dh_dt = h + x
    return coeff, dx_dt, dh_dt
```


# Hints

Consider the following hints:
* Only return the `def get_equations(t, x, h, mode, nl_params)` function, put your thinkings and explanations in the docstring of the function or as comments in the code (between `def` and `return`).
* Only use valid `nd2py` expressions. Do not use numpy or any other libraries that are not part of ND2py. Do not index or slice `x` since it is a unsubscriptable object. Do not use Python's built-in functions like `sum`, `map`, or `filter` to perform operations on the state variables.
* Only use `nl_params[i]` to define time-varying coefficients. Do not introduce symbolic placeholders like `a_i` or `c_ij`.
* Leverage domain knowledge about the coupled dynamics of SST and WWV to generate the equations, such as the coupling strength of interannual variations or the asymmetry of ENSO.
* `sour(...)` can be understand as the variable of other nodes, while `targ(...)` can be understand as the variable of the current node. `aggr(...)` can be understand as the aggregation of the incoming edges of the current node. Therefore, `sour` and `targ` can only appear within `aggr`.
* ND2py provides common mathematical functions, including `sin`, `cos`, `exp`, `logabs`, `inv`, `neg`, `sqrtabs`, `pow2`, `pow3`, `tanh`, `sigmoid`, etc., which can be used to express complex nonlinear relationships between the state variables of the nodes. ND2py also provides basic arithmetic operations such as `+`, `-`, `*`, `/`, and `**` for combining these functions.
* Don’t limit yourself to linear terms—consider using `tanh`, `pow2`, `inv`, or other nonlinear forms to reflect threshold behavior, saturation, or asymmetry.
* Optionally, explore models where the influence of incoming signals (via `aggr` and `readout`) is non-additive, for example using `tanh(aggr(...))`, or modulated by current node state (e.g., `x * aggr(...)`).
* You may see demo programs, and they may violate the above instructions, such as not adding instructions in the docstring, or using unnecessary `nl_params`. You should only refer to the items they use for their physical meaning, and do not imitate them where they violate the instructions.