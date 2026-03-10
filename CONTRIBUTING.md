# Contributing to ShrinkageTrees

Thank you for considering contributing to `ShrinkageTrees`! Contributions of all
kinds are welcome, whether they are bug reports, code improvements, new features,
or documentation updates.

## How to contribute

- **Report bugs or request features**  
  Please use the [GitHub Issues](https://github.com/tijn-jacobs/ShrinkageTrees/issues)
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

## Extending priors

The C++ backend of `ShrinkageTrees` is designed to make it straightforward to
add new global–local shrinkage priors. Priors are managed by the `ScaleMixture`
class, which wraps objects derived from the virtual base class `EtaPrior`.
This design provides a consistent interface for proposing step heights,
computing log densities, updating parameters, and handling global vs. local
shrinkage effects.

Currently implemented priors include `FixedVariance`, `HalfCauchy`,
`Horseshoe`, and `Horseshoe_fw`. To extend the package with a new prior, you
can create a new class that inherits from `EtaPrior` and implements the core
methods:

- `Propose()` — generate new step height and prior parameter values  
- `GlobalUpdate()` — update global shrinkage parameters, if applicable  
- `LogProposeDensity()` — compute log-density of the proposal  
- `LogPrior()` — compute the prior log-density  
- `GetVariance()` — return the variance implied by the prior  
- `GetGlobal()` — indicate if global parameters are required  

New priors can then be registered via the `CreateEtaPrior()` factory function,
which allows them to be called from the R interface without altering existing
code. The modular design ensures that additional priors integrate seamlessly
into the reversible jump MCMC framework used in the package.

Contributions in this direction are especially welcome.

## Code of Conduct
By participating in this project you agree to follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).

## Questions?
If you have questions about using or contributing to `FusionForests`, please
open a discussion or issue on GitHub.
