#ifndef GUARD_Tree_h
#define GUARD_Tree_h

// This class represents a node in a binary decision tree, where each node is 
// connected to other nodes through parent, left, and right pointers. The tree 
// structure allows for recursive data splitting based on specified cutpoints, 
// which are stored in an external `Cutpoints` object.
//
// Each node stores:
// - `parameter`: An object containing all the node specific and, possibly,  
//                global parameters.
// - `split_var`: The index of the variable used for splitting at this node.
// - `cut_val`: The index of the cutpoint in `Cutpoints` used for splitting.
//
// The tree can grow by adding children to nodes or prune by removing them. 
// Utility functions allow for manipulating the tree, collecting nodes, and 
// evaluating the tree's performance on given data.

#include "ScaleMixture.h"

class Tree {

private:

  // Parameters class replaces the univariate parameter; can represent multiple 
  // parameters for a node. Depends on choice of prior for the step heights.
  Parameters parameters;  

  // Splitting information: 
  // The observation is assigned to the left node if x[split_var] < 
  // Cutpoints.values[split_var][cut_val].
  size_t split_var;
  size_t cut_val;

  // Pointers for the tree structure.
  Tree* parent; 
  Tree* left; 
  Tree* right; 

  // ?
  size_t leaf_index_;  // valid iff this is a leaf during current stats pass


  // Utility function to copy the structure and data from another tree. 
  // This function might be removed if not needed (indicated by the DELETE 
  // comment).
  void CopyTree(Tree* n, const Tree* o); // DELETE?

public:

  // Constructors and Destructor

  // Default constructor with (unused) pointer
  Tree(double* common_paramaters)
    : parameters(),
      split_var(0),
      cut_val(0),
      parent(nullptr),
      left(nullptr),
      right(nullptr),
      leaf_index_(std::numeric_limits<size_t>::max()) {}

  // Plain default constructor
  Tree()
    : parameters(),
      split_var(0),
      cut_val(0),
      parent(nullptr),
      left(nullptr),
      right(nullptr),
      leaf_index_(std::numeric_limits<size_t>::max()) {}

  // Copy constructor
  Tree(const Tree& new_tree)
    : parameters(),
      split_var(0),
      cut_val(0),
      parent(nullptr),
      left(nullptr),
      right(nullptr),
      leaf_index_(std::numeric_limits<size_t>::max()) {
    CopyTree(this, &new_tree);
  }

  // Destructor that reduces the tree to a single node (the root).
  void CutDownTree();
  ~Tree() { CutDownTree(); }

  // Setters and Getters

  // Set all parameters at once using a pointer to an array
  void SetParameters(const double* new_params) {
    parameters.SetParameters(new_params);
  }

  // Set a specific parameter at an index
  void SetParameters(size_t index, double new_param) {
    parameters.SetParameters(index, new_param);
  }

  // Set all parameters using a Parameters object
  void SetParameters(const Parameters& params) {
    parameters = params; // Uses the copy assignment operator of the Parameters 
                         // class
  }

 // Get a specific parameter at an index
  double GetParameters(size_t index) const {
    return parameters.GetParameters(index);
  }

   // Get a specific parameter at an index
  double GetGlobalParameters(size_t index) const {
    return parameters.GetGlobalParameters(index);
  }

  // Set a specific parameter at an index
  void SetGlobalParameters(size_t index, double new_param) {
    parameters.SetGlobalParameters(index, new_param);
  }

  // Get a reference to the Parameters object
  Parameters& GetParameters() {
    return parameters; // Return a reference, not a temporary object
  }

  void SetSplitVar(size_t _split_var) { split_var = _split_var; }
  void SetCutVal(size_t _cut_val) { cut_val = _cut_val; }
  size_t GetSplitVar() const { return split_var; }
  size_t GetCutVal() const { return cut_val; }
  Tree* GetParent() const { return parent; }  
  Tree* GetLeft() const { return left; }
  Tree* GetRight() const { return right; }

  // Tree Manipulation Functions

  // Get a pointer to the node with the given ID. Returns nullptr if the node is 
  // not found.
  Tree* GetNodePointer(size_t ID); 

  // Print the tree or a single node. If also_print_children is true, the entire 
  // tree is printed.
  void PrintTree(bool also_print_children = true) const;

  // Return the total number of nodes in the tree.
  size_t TreeSize() const;

  // Count the number of "nog" nodes in the tree (nodes with no grandchildren).
  size_t NumberOfNogs() const;

  // Count the number of leaf nodes (terminal nodes) in the tree.
  size_t NumberOfLeaves() const;

  // Add children to a terminal node.
  void GrowChildren(Tree* leaf, size_t _split_var, size_t _cut_val, 
                    Parameters par_left, Parameters par_right); 

  // Remove the children of a nog node by ID, updating the node's parameters.
  bool KillChildren(size_t node_ID, Parameters parameter);

  // Remove the children of a nog node, updating the node's parameter value.
  void KillChildren(Tree* nog_node, Parameters parameter);

  // Remove the children of a nog node, assumes that the node's parameters have 
  // been updated.
  void KillChildren(Tree* nog_node);

  // Collect pointers to all leaf nodes in the tree.
  void CollectLeaves(std::vector<Tree*>& leaf_vector);

  // Collect pointers to all "nog" nodes in the tree.
  void CollectNogs(std::vector<Tree*>& nog_vector);

  // Collect pointers to all nodes in the tree.
  void CollectNodes(std::vector<Tree*>& node_vector); // DELETE?

  // Collect pointers to all nodes in the tree (const version).
  void CollectNodes(std::vector<const Tree*>& node_vector) const;

  inline Tree* FindLeaf(double* x_row,
                            const Cutpoints& cutpoints) const noexcept {

    const Tree* node = this;

    while (node->left) {
      node = (x_row[node->split_var] <
              cutpoints.values[node->split_var][node->cut_val])
               ? node->left
               : node->right;
    }

    return const_cast<Tree*>(node);
  }

  // Determine the possible cuts for a given variable, adjusting the interval 
  // [lower_bound, upper_bound].
  void PossibleCuts(size_t split_var, int* lower_bound, int* upper_bound) const;

  // Evaluate the tree on a given set of covariates, storing the results in 
  // estimated_outcomes.
  void EvaluateTree(Cutpoints& cutpoints, size_t p, size_t n, double* x, 
                    Data& data, double* estimated_outcomes);

  // Method to print the structure of the full tree
  void PrintFullTree(int depth, Cutpoints& cutpoints, Data& data);

  // Node Information Functions

  // Return the ID of the node, determined based on its position in the tree.
  size_t NodeID() const; 

  // Compute the depth of the node in the tree.
  size_t NodeDepth() const;

  // Check whether the node is a "nog" (a node with exactly two descendants).
  bool IsNog() const;

  // Recursively find the cut value for a given split variable.
  size_t FindSameCut(size_t split_var) const;

  // Returns the number of observations that fall into this node.
  size_t NodeSize(Data& data, Cutpoints& cutpoints);

  // Funcions to manage leaf_index_
  inline void   ResetLeafIndex()         { leaf_index_ = std::numeric_limits<size_t>::max(); }
  inline void   SetLeafIndex(size_t idx) { leaf_index_ = idx; }
  inline size_t GetLeafIndex() const     { return leaf_index_; }
  inline bool   HasLeafIndex() const     { return leaf_index_ != std::numeric_limits<size_t>::max(); }

};

#endif // GUARD_Tree_h

