#include "Parameters.h"

// Constructor initializes node-specific and global parameters
Parameters::Parameters() {
  parameters[0] = 0.0;
  parameters[1] = 0.1;
  parameters[2] = 0.1;

  global_parameters[0] = 1.0;
  global_parameters[1] = 1.0;
}

// Resets node-specific and global parameters to default values
void Parameters::Reset() {
  parameters[0] = 0.0;
  parameters[1] = 0.1;
  parameters[2] = 0.1;

  global_parameters[0] = 1.0;
  global_parameters[1] = 1.0;
}

// Assignment operator to copy node-specific and global parameters
Parameters& Parameters::operator=(const Parameters& other) {
  if (this != &other) {
    std::copy(other.parameters,
              other.parameters + number_of_parameters,
              parameters);
    std::copy(other.global_parameters,
              other.global_parameters + number_of_global_parameters,
              global_parameters);
  }
  return *this;
}

// Sets all node-specific parameters using a pointer to an array
void Parameters::SetParameters(const double* new_parameters) {
  std::copy(new_parameters, new_parameters + number_of_parameters, parameters);
}

// Sets a node-specific parameter at a given index
void Parameters::SetParameters(size_t index, double new_parameter) {
  parameters[index] = new_parameter;
}

// Sets a global parameter at a given index
void Parameters::SetGlobalParameters(size_t index, double new_parameter) {
  global_parameters[index] = new_parameter;
}

// Returns all node-specific parameters (pointer to array)
const double* Parameters::GetAllParameters() const {
  return parameters;
}

// Retrieves a node-specific parameter at a given index
double Parameters::GetParameters(size_t index) const {
  return parameters[index];
}

// Retrieves a global parameter at a given index
double Parameters::GetGlobalParameters(size_t index) const {
  return global_parameters[index];
}

// Retrieves the total number of node-specific parameters
size_t Parameters::GetNumberOfParameters() const {
  return number_of_parameters;
}
