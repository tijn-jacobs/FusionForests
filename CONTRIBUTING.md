# Contributing to FusionForests

Thank you for considering contributing to `FusionForests`! Contributions of
all kinds are welcome, whether they are bug reports, code improvements, new
features, or documentation updates.

## How to contribute

- **Report bugs or request features**  
  Please use the [GitHub Issues](https://github.com/tijn-jacobs/FusionForests/issues)
  page to report problems or suggest improvements. When reporting a bug, try to
  include a reproducible example.

- **Contribute code**  
  1. Fork the repository.  
  2. Create a new branch for your feature or fix.  
  3. Make your changes, and ensure that all tests pass using:  
     ```r
     devtools::check()
     ```  
  4. Submit a pull request with a clear description of your changes.

- **Coding style**  
  - Follow the existing R and C++ style in the package.  
  - Keep lines under 80 characters where possible.  
  - Add comments where code is non-obvious.  
  - Please include tests or examples for new functionality.

- **Documentation**  
  If you add or change functions, update the documentation accordingly using
  roxygen2 comments (`#'`). Documentation contributions are as valuable as code.

## Package structure

The MCMC sampler is implemented in C++ (under `src/`) and interfaced via
Rcpp. The forest machinery (`StanForest`, `StanTree`, `StanBirthDeath`)
implements standard BART and Dirichlet BART (DART) tree updates; the entry
points `FusionForest.cpp`, `SimpleBART.cpp` and `SimpleBCF.cpp` assemble
these into the exported models, with the commensurate prior for information
borrowing in `CommensurateParameters.cpp`.

The single-study causal and survival models re-exported by this package are
developed in the [ShrinkageTrees](https://github.com/tijn-jacobs/ShrinkageTrees)
package; contributions to those models are best made there.
