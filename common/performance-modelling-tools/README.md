# Performance Modeling Tools

This repository hosts configuration files and scripts to support performance modeling of research application on Pawsey Supercomputing Centre systems. 

We have provided 
* [Spack environments](./spack) that provide performance modeling toolchains for Pawsey systems
* [Examples](./examples/) that illustrate how to use this repository for active research projects
* [Scripts](./bin/) for processing profiler data to create graphics and other post-processed data to support performance modeling

## About this Repository

### Scaffolding
This repository is organized with the following top-level directories

* `bin/` : Contains scripts for post-processing profiler data
* `docs/` : Contains mkdocs documentation
* `examples/` : Contains example configurations for creating profile data and processing output with the tools in this repository.
* `spack/` : Contains spack environments for various Pawsey systems that provide hpc-toolkit and other useful tools.
* `tests/`: Contains some simple tests for exploring device offloading with HIP, CUDA, OpenMP. Currently has a single test, `warmup/`


## Getting Help
Currently, you can request help by [submitting an issue](https://github.com/PawseySC/performance-modelling-tools/issues)



