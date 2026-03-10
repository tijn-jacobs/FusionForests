#ifndef GUARD_StanTree_h
#define GUARD_StanTree_h

#include "Prerequisites.h"

// CutpointMatrix[v][c] is the c-th cutpoint for predictor v.
// An observation goes left when x[v] < CutpointMatrix[v][c].
using CutpointMatrix = std::vector<std::vector<double>>;

// Information about a single node, used by the input-stream operator.
struct NodeInfo {
  std::size_t id;
  std::size_t split_var;
  std::size_t cut_val;
  double step_height;
};


// A node in a binary regression tree carrying a scalar leaf parameter
// (step_height).  Each node stores its own split rule (split_var, cut_val),
// and pointers to its parent, left child, and right child.
class StanTree {
public:
  // Stream operator for deserialisation
  friend std::istream& operator>>(std::istream&, StanTree&);

  // Constructors and destructor
  StanTree()
    : step_height(0.0), split_var(0), cut_val(0),
      parent(nullptr), left(nullptr), right(nullptr) {}
  StanTree(const StanTree& other)
    : step_height(0.0), split_var(0), cut_val(0),
      parent(nullptr), left(nullptr), right(nullptr) {
    CopyTree(this, &other);
  }
  explicit StanTree(double initial_step_height)
    : step_height(initial_step_height), split_var(0), cut_val(0),
      parent(nullptr), left(nullptr), right(nullptr) {}
  void Clear(); // Reduce the tree back to a single root node
  ~StanTree() { Clear(); }

  // Assignment operator
  StanTree& operator=(const StanTree&);

  // Setters
  void SetStepHeight(double value) { step_height = value; }
  void SetSplitVar(size_t var)     { split_var = var; }
  void SetCutVal(size_t val)       { cut_val = val; }

  // Getters
  double GetStepHeight() const { return step_height; }
  size_t GetSplitVar()   const { return split_var; }
  size_t GetCutVal()     const { return cut_val; }
  StanTree* GetParent()        { return parent; }
  StanTree* GetLeft()          { return left; }
  StanTree* GetRight()         { return right; }

  // Return a pointer to the node with the given id (nullptr if not found).
  StanTree* GetNodePointer(size_t id);

  // Find the leaf node reached by observation x using the given cutpoints.
  StanTree* FindLeaf(double* x, CutpointMatrix& cutpoints);

  // Recursively narrow the valid range [lower_bound, upper_bound] for
  // split_var by walking up to the root.
  void FindRegionBounds(size_t split_var, int* lower_bound, int* upper_bound);

  // Return the cut value imposed by the nearest ancestor that splits on
  // split_var (used for the degenerate-tree augmentation strategy).
  size_t GetConstrainedCut(size_t split_var);

  // Node type queries
  size_t NodeID()    const; // 1-based binary-heap index
  size_t NodeDepth();       // Distance from root (0 = root)
  char   NodeType();        // 't'=top 'b'=bottom 'n'=no-grandchildren 'i'=interior
  bool   IsNog();           // True iff node has children but no grandchildren

  // Tree metrics
  size_t TreeSize();        // Total number of nodes
  size_t NumberOfNogs();    // Number of no-grandchildren nodes
  size_t NumberOfLeaves();  // Number of leaf (bottom) nodes

  // Grow a leaf into an internal node (addressed by node id).
  bool Birth(size_t nid, size_t split_var, size_t cut_val,
             double left_step_height, double right_step_height);
  // Remove the two leaf children of a nog node (addressed by node id).
  bool Death(size_t nid, double new_step_height);
  // Grow a leaf into an internal node via direct pointer.
  void BirthAtNode(StanTree* leaf, size_t split_var, size_t cut_val,
                   double left_step_height, double right_step_height);
  // Remove children of a nog node via direct pointer.
  void DeathAtNode(StanTree* nog_node, double new_step_height);

  // Collect pointers to all leaf nodes.
  void CollectLeaves(std::vector<StanTree*>& leaf_vector);
  // Collect pointers to all no-grandchildren nodes.
  void CollectNogs(std::vector<StanTree*>& nog_vector);
  // Collect pointers to all nodes.
  void CollectNodes(std::vector<StanTree*>& node_vector);
  void CollectNodes(std::vector<const StanTree*>& node_vector) const;

  // Print the tree to screen.
  void Print(bool also_print_children = true);

private:
  double step_height; // Leaf parameter (step height)
  // Split rule: go left when x[split_var] < cutpoints[split_var][cut_val]
  size_t split_var;
  size_t cut_val;
  // Tree structure pointers
  StanTree* parent;
  StanTree* left;
  StanTree* right;

  // Deep-copy the tree rooted at source into destination (destination must
  // have no children).
  void CopyTree(StanTree* destination, const StanTree* source);
};

#endif

