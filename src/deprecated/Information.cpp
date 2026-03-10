#include "Information.h"


// Function to print the data object
void Data::Print() const {
  // Print general information about the data object
  cout << "Data Object:" << endl;
  cout << "Number of variables (p): " << p << endl;
  cout << "Number of observations (n): " << n << endl;

  // Print covariate (X) data for each observation
  cout << "X (covariates):" << endl;
  for (size_t i = 0; i < n; ++i) {
    cout << "Observation " << i + 1 << ": ";
    for (size_t j = 0; j < p; ++j) {
      cout << X[i * p + j] << " ";  // Print each covariate value
    }
    cout << endl;
  }

  // Print outcome (y) data for each observation
  cout << "y (outcomes):" << endl;
  for (size_t i = 0; i < n; ++i) {
    cout << "Observation " << i + 1 << ": " << y[i] << endl;
  }
}


// Function to print the prior parameters
void TreePrior::Print() const {
  
  // Print header for the prior parameters
  cout << "----------------------------------" << std::endl;
  cout << "         Prior Parameters         " << std::endl;
  cout << "----------------------------------" << std::endl;

  // Print each prior parameter with a fixed-width label and value format
  cout << std::left << std::setw(15) << "p_GROW:"    
            << std::setw(10) << p_GROW   << std::endl;
  cout << std::left << std::setw(15) << "p_PRUNE:"   
            << std::setw(10) << p_PRUNE  << std::endl;
  cout << std::left << std::setw(15) << "base:"      
            << std::setw(10) << base     << std::endl;
  cout << std::left << std::setw(15) << "power:"     
            << std::setw(10) << power    << std::endl;
  cout << std::left << std::setw(15) << "eta:"       
            << std::setw(10) << eta      << std::endl;

  // Print footer
  cout << "----------------------------------" << std::endl;
}


// number_of_cuts is a p-dimensional vector containing the number of cuts that 
// need to be added to the cutpoint matrix for each variable. 
// Note that this usually equals the number of observations (assuming continuous 
// covariates). In other cases, it may be the number of unique values of a 
// certain covariate.
void Cutpoints::SetCutpoints(size_t _p, size_t n, double* X, int* number_of_cuts) {

  // Set the number of covariates and resize the cutpoints matrix
  p = _p;
  values.resize(p);

  // If no no. of cutpoints are specified, we use the emperical values
  if (number_of_cuts != nullptr) {

    // Create a vector of the number of cuts for each covariate
    std::vector<int> cuts(p);
    for (size_t i = 0; i < p; i++) cuts[i] = number_of_cuts[i];

    // Initialize vectors for the minimum and maximum values of each covariate
    std::vector<double> min_values(p, std::numeric_limits<double>::infinity());
    std::vector<double> max_values(p, -std::numeric_limits<double>::infinity());

    // Compute the min and max values for each covariate
    for (size_t i = 0; i < p; ++i) {
      for (size_t j = 0; j < n; ++j) {
        double value = *(X + p * j + i);
        if (value < min_values[i]) min_values[i] = value;
        if (value > max_values[i]) max_values[i] = value;
      }
    }

    // Set the cutpoints based on the number of cuts
    for (size_t i = 0; i < p; ++i) {
      // Calculate interval size (delta) between cutpoints for the i-th variable
      double delta = (max_values[i] - min_values[i]) / (cuts[i] + 1.0);
      
      // Resize the vector to hold the cutpoints for the i-th variable
      values[i].resize(cuts[i]);

      // Initialize counter for cutpoint indexing
      size_t j = 0;

      // Iterate over cutpoints and set their values based on min_values and delta
      for (double& cutpoint : values[i]) {
        cutpoint = min_values[i] + (++j) * delta;
      }
    }

  } else {
    
    // If number_of_cuts is not specified, use unique sorted covariate values
    for (size_t i = 0; i < p; ++i) {
      // Extract observed values for the i-th variable
      std::vector<double> observed_values(n);
      for (size_t j = 0; j < n; ++j) {
        observed_values[j] = *(X + p * j + i);
      }

      // Sort the observed values
      std::sort(observed_values.begin(), observed_values.end());

      // Remove duplicates to get unique cutpoints
      auto last = std::unique(observed_values.begin(), observed_values.end());
      observed_values.erase(last, observed_values.end());

      // Resize the vector to hold the unique sorted cutpoints
      values[i] = observed_values;
    }
  }
}


// Function to print the cutpoints
void Cutpoints::Print() const {
  // Print header for the cutpoints information
  cout << "Cutpoints Information:\n";
  cout << "-----------------------\n";

  // Loop over each variable and print its cutpoints
  for (size_t i = 0; i < p; ++i) {
    cout << "Variable " << i << " cutpoints: ";

    // Print each cutpoint value with fixed precision
    for (size_t j = 0; j < values[i].size(); ++j) {
      cout << std::fixed << std::setprecision(3) 
                << values[i][j] << " ";
    }

    // New line after printing all cutpoints for the variable
    cout << std::endl;
  }

  // Print footer
  cout << "-----------------------\n";
}

