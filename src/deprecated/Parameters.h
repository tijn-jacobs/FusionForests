// The Parameters class stores both node-specific and common parameters used  
// within the tree structure. The node-specific parameters are unique to each  
// node and can be modified independently, while the common parameters are  
// shared across nodes to ensure consistency and synchronization within the  
// tree. The class allows for setting, retrieving, and updating these  
// parameters efficiently. Additionally, it provides flexibility for managing  
// common parameters via pointers, enabling changes in one part of the tree  
// to be reflected in all nodes sharing those parameters. This structure helps  
// facilitate the manipulation of parameters during the MCMC process,  
// particularly when performing operations such as proposing new values or  
// updating based on observed data.

#ifndef GUARD_Parameters_h
#define GUARD_Parameters_h

#include "Prerequisites.h"

class Parameters {
  static const size_t number_of_parameters = 3;  // Fixed number of node-specific
                                                 // parameters
  double parameters[number_of_parameters];  // Node-specific parameter values

  static const size_t number_of_global_parameters = 2;  // Fixed number of global
                                                        // parameters
  double global_parameters[number_of_global_parameters];  // Global parameter
                                                          // values

 public:
  // Constructor initializes node-specific and global parameters
  Parameters();

  // Resets node-specific and global parameters to default values
  void Reset();

  // Assignment operator to copy node-specific and global parameters
  Parameters& operator=(const Parameters& other);

  // Sets all node-specific parameters using a pointer to an array
  void SetParameters(const double* new_parameters);

  // Sets a node-specific parameter at a given index
  void SetParameters(size_t index, double new_parameter);

  // Sets a global parameter at a given index
  void SetGlobalParameters(size_t index, double new_parameter);

  // Returns all node-specific parameters (pointer to array)
  const double* GetAllParameters() const;

  // Retrieves a node-specific parameter at a given index
  double GetParameters(size_t index) const;

  // Retrieves a global parameter at a given index
  double GetGlobalParameters(size_t index) const;

  // Retrieves the total number of node-specific parameters
  size_t GetNumberOfParameters() const;
};

#endif  // GUARD_Parameters_h
