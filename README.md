# Compromise.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://manuelbb-upb.github.io/Compromise.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://manuelbb-upb.github.io/Compromise.jl/dev/)
[![Build Status](https://github.com/manuelbb-upb/Compromise.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/manuelbb-upb/Compromise.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/manuelbb-upb/Compromise.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/manuelbb-upb/Compromise.jl)

## CoMPrOMISE
**Co**nstrained **M**ultiobjective **Pr**oblem **O**ptimizer with **M**odel **I**nformation to **S**ave **E**valuations

This package provides a fExible first-order solver for constrained and unconstrained nonlinear multiobjective problems.
It uses a trust region approach and either exact derivatives or local surrogate models for a derivative-free descent.
Box constraints are respected during model construction and treated as unrelaxable.
Box constraints and linear constraints are supported and passed down to an inner LP solver.
Nonlinear constraint functions can be modelled and are dealt with by incorporating a filter. 
They are *relaxable*, i.e., all other functions must be evaluable even when the constraints are violated.

