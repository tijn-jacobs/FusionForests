#ifndef COMMENSURATE_PARAMETERS_H
#define COMMENSURATE_PARAMETERS_H

#include "Prerequisites.h"
#include <cmath>

class CommensurateParameters {

private:
    double eta;       // commensurate shift
    double nu;        // precision
    double rho;       // spike probability
    int    w;         // spike/slab indicator
    double spike;    // fixed spike precision
    double B1, B2;    // slab Uniform bounds
    double a0, b0;    // Beta prior hyperparameters for rho

public:

    // Constructor
    CommensurateParameters(double eta_init,
                           double nu_init,
                           double rho_init,
                           int w_init,
                           double spike_precision,
                           double B1_in,
                           double B2_in,
                           double a_prior,
                           double b_prior)
        : eta(eta_init),
          nu(nu_init),
          rho(rho_init),
          w(w_init),
          spike(spike_precision),
          B1(B1_in),
          B2(B2_in),
          a0(a_prior),
          b0(b_prior) {}

    // ======== Accessors ========
    double getEta() const { return eta; }
    double getnu() const { return nu; }
    int    getW()   const { return w;   }
    double getRho()   const { return rho;   }

    // ======== Main update routines ========
    void updateEta(double R, int n0, double sigma, Random &rng);
    void updateSpikeSlab(Random &rng);
    void updateRho(Random &rng);

};

#endif
