// This file contains the class definitions of four important objects:  
// - Data:       contains the observed data and the residuals that are  
//               updated in each step. The covariate data X is stored in  
//               contiguous rows, that is, in an array of length n*p. The  
//               j-th covariate of person i can be retrieved by X[i * p + j].  
//               This class represents the dataset used for analysis,  
//               consisting of features (X), outcomes (y), and residuals.  
//               It provides functionality for accessing and manipulating  
//               the dataset, and supports retrieving specific rows of data,  
//               outcomes, and residuals.  
// - TreePrior:  stores prior information used in MCMC (Markov Chain  
//               Monte Carlo) methods, including probabilities for birth/  
//               death moves in decision trees and parameters that control  
//               the tree's shape and complexity. This class holds the prior  
//               probabilities and parameters such as base, power, and eta,  
//               which define the prior on the tree topologies.  
// - Cutpoints:  manages the cutpoints used in decision tree algorithms.  
//               Cutpoints determine the thresholds at which the data is  
//               split for each covariate. This class handles the generation,  
//               storage, and retrieval of these cutpoints, which are  
//               crucial for defining the decision boundaries in the tree.  

#ifndef GUARD_Information_h
#define GUARD_Information_h

#include "Parameters.h"

class Data {
  size_t p;   // Number of variables
  size_t n;   // Number of observations
  double* X;  // Pointer to contiguous row array of data (n x p)
  double* y;  // Pointer to array with outcomes (n x 1)
  double* weights;  // Per-observation weights for weighted regression (nullptr = all 1.0)

 public:
  double* residual;  // Pointer to array of residuals (n x 1)

  // Default constructor initializing data pointers to nullptr
  Data() : p(0), n(0), X(nullptr), y(nullptr), weights(nullptr), residual(nullptr) {}

  // Constructor that initializes with given dimensions and pointers for X and y
  Data(size_t _p, size_t _n, double* _X, double* _y)
      : p(_p), n(_n), X(_X), y(_y), weights(nullptr), residual(nullptr) {}

  // Getter and Setter for the number of variables (p)
  size_t GetP() const { return p; }
  void SetP(size_t _p) { p = _p; }

  // Getter and Setter for the number of observations (n)
  size_t GetN() const { return n; }
  void SetN(size_t _n) { n = _n; }

  // Getter and Setter for the pointer to feature data (X)
  double* GetX() const { return X; }
  void SetX(double* _X) { X = _X; }

  // Getter and Setter for the pointer to outcome data (y)
  double* GetY() const { return y; }
  void SetY(double* _y) { y = _y; }

  // Getter and Setter for the pointer to residual data
  double* GetResidual() const { return residual; }
  void SetResidual(double* _residual) { residual = _residual; }

  // Retrieve a specific row of data (covariates for a specific observation)
  double* GetDataRow(int i) const { return X + i * p; }

  // Retrieve the outcome for a specific observation
  double GetOutcome(size_t i) const { return y[i]; }

  // Retrieve the residual for a specific observation
  double GetResidual(size_t i) const { return residual[i]; }

  // Per-observation weights for weighted regression (e.g., b_i^2 in BCF)
  void SetWeights(double* w) { weights = w; }
  double* GetWeights() const { return weights; }
  bool HasWeights() const { return weights != nullptr; }
  double GetWeight(size_t i) const { return weights ? weights[i] : 1.0; }

  // Function to print the contents of the data object
  void Print() const;

  // Temporary for optimization
  inline const double* GetDataRowConst(size_t i) const noexcept {
    return X + i * p; // assuming row-major like you already use
  }

};

class TreePrior {
 public:
  // Default constructor initializing prior parameters for tree moves and 
  // structure
  TreePrior()
      : p_GROW(0.4), p_PRUNE(0.4),  // Default probabilities for tree moves
        base(0.95), power(2.0), eta(1.0) {}  // Default topology parameters

  // Constructor with custom parameters for tree moves and structure
  TreePrior(double _p_GROW, double _p_PRUNE, double _base, double _power,
            double _eta)
      : p_GROW(_p_GROW), p_PRUNE(_p_PRUNE),  // Custom tree move probabilities
        base(_base), power(_power), eta(_eta) {}  // Custom topology parameters

  // Prior probabilities for tree moves
  double p_GROW;   // Probability of growing a tree
  double p_PRUNE;  // Probability of pruning a tree

  // Prior parameters for tree topology
  double base;     // Base parameter for tree growth
  double power;    // Power parameter for tree growth

  // Prior variance of the step height; only used in non-reversible steps
  double eta;

  // Function to print the prior parameters
  void Print() const;
};

class Cutpoints {
 public:

  // Vector to store cutpoints for each variable
  std::vector<std::vector<double>> values;

  // Number of covariates (features) for which cutpoints are generated
  size_t p;

  // Flag to determine whether the cutpoints are uniform or based on the data
  bool uniform;

  // Default constructor initializes the values vector, sets p to 0, and uniform
  // to false
  Cutpoints() : values(), p(0), uniform(false) {}

  // Function to set cutpoints based on the number of cuts and observed data
  void SetCutpoints(size_t p, size_t n, double* X, int* number_of_cuts);

  // Function to print the cutpoints to the console
  void Print() const;
};

#endif // GUARD_Information_h
