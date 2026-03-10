#include "CommensurateParameters.h"
#include <cmath>

// 1. Update eta | rest   (Gibbs update)
void CommensurateParameters::updateEta(double residual, int n0, double sigma, Random &rng) {

    // posterior precision of eta
    double prec = nu * nu + n0 / (sigma * sigma);
    double var  = 1.0 / prec;
    double mean = var * (residual / (sigma * sigma));

    // Draw eta ~ Normal(mean, var)
    eta = mean + std::sqrt(var) * rng.normal();
}


// Update nu and w   (spike/slab update)
void CommensurateParameters::updateSpikeSlab(Random &rng) {

    // Draw slab precision u ~ Uniform(B1, B2)
    double u = B1 + (B2 - B1) * rng.uniform();

    // Compute Normal densities N(η | 0, ν = spike) and N(η | 0, ν = u)
    // Precision = ν^2 -> variance = 1/ν^2
    double d_spike = std::exp(-0.5 * eta * eta * (spike * spike));
    double d_slab  = std::exp(-0.5 * eta * eta * (u     * u));

    // Mixture weights
    double w1 = rho * d_spike;
    double w0 = (1.0 - rho) * d_slab;
    double prob_spike = w1 / (w1 + w0);

    // Sample w (indicator for spike)
    double u_rand = rng.uniform();
    w = (u_rand < prob_spike ? 1 : 0);

    // Assign nu
    nu = (w == 1 ? spike : u);
}


// ===========================================================
// 3. Update rho | w   (optional Beta update)
// ===========================================================
void CommensurateParameters::updateRho(Random &rng) {

    double alpha = a0 + w;
    double beta  = b0 + (1 - w);

    // Beta(a,b) via Gamma(a)/[Gamma(a)+Gamma(b)]
    double g1 = rng.gamma(alpha, 1.0);
    double g2 = rng.gamma(beta,  1.0);

    rho = g1 / (g1 + g2);
}
