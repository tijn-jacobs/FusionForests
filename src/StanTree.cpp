#include "StanTree.h"

// Return the 1-based binary-heap node id.
size_t StanTree::NodeID() const
{
  if (!parent) return 1; // Root has no parent
  if (this == parent->left) return 2 * (parent->NodeID());
  else return 2 * (parent->NodeID()) + 1;
}

// Return a pointer to the node with the given id, or nullptr if not found.
StanTree* StanTree::GetNodePointer(size_t id)
{
  if (this->NodeID() == id) return this;
  if (left == nullptr) return nullptr;
  StanTree* found_left = left->GetNodePointer(id);
  if (found_left) return found_left;
  StanTree* found_right = right->GetNodePointer(id);
  if (found_right) return found_right;
  return nullptr;
}

// Grow a leaf (addressed by node id) into an internal node.
bool StanTree::Birth(size_t nid, size_t split_var, size_t cut_val,
                     double left_step_height, double right_step_height)
{
  StanTree* target = GetNodePointer(nid);
  if (target == nullptr) {
    cout << "error in Birth: bottom node not found\n";
    return false;
  }
  if (target->left != nullptr) {
    cout << "error in Birth: found node has children\n";
    return false;
  }

  StanTree* left_child  = new StanTree;
  left_child->step_height  = left_step_height;
  StanTree* right_child = new StanTree;
  right_child->step_height = right_step_height;

  target->left      = left_child;
  target->right     = right_child;
  target->split_var = split_var;
  target->cut_val   = cut_val;
  left_child->parent  = target;
  right_child->parent = target;

  return true;
}

// Return the depth of this node (root = 0).
size_t StanTree::NodeDepth()
{
  if (!parent) return 0;
  return 1 + parent->NodeDepth();
}

// Return the total number of nodes in the tree.
size_t StanTree::TreeSize()
{
  if (left == nullptr) return 1;
  return 1 + left->TreeSize() + right->TreeSize();
}

// Return the node type character: t=top, b=bottom, n=no-grandchildren, i=interior.
char StanTree::NodeType()
{
  if (!parent) return 't';
  if (!left)   return 'b';
  if (!(left->left) && !(right->left)) return 'n';
  return 'i';
}

// Remove the children of the nog node with the given id.
bool StanTree::Death(size_t nid, double new_step_height)
{
  StanTree* nog_node = GetNodePointer(nid);
  if (nog_node == nullptr) {
    cout << "error in Death: nid invalid\n";
    return false;
  }
  if (nog_node->IsNog()) {
    delete nog_node->left;
    delete nog_node->right;
    nog_node->left        = nullptr;
    nog_node->right       = nullptr;
    nog_node->split_var   = 0;
    nog_node->cut_val     = 0;
    nog_node->step_height = new_step_height;
    return true;
  } else {
    cout << "error in Death: node is not a nog node\n";
    return false;
  }
}

// Return true iff this node has children but no grandchildren.
bool StanTree::IsNog()
{
  bool is_nog = true;
  if (left) {
    if (left->left || right->left) is_nog = false;
  } else {
    is_nog = false; // Leaf node has no children
  }
  return is_nog;
}

size_t StanTree::NumberOfNogs()
{
  if (!left) return 0;
  if (left->left || right->left) {
    return left->NumberOfNogs() + right->NumberOfNogs();
  } else {
    return 1;
  }
}

size_t StanTree::NumberOfLeaves()
{
  if (left == nullptr) return 1;
  return left->NumberOfLeaves() + right->NumberOfLeaves();
}

// Collect pointers to all leaf nodes.
void StanTree::CollectLeaves(std::vector<StanTree*>& leaf_vector)
{
  if (left) {
    left->CollectLeaves(leaf_vector);
    right->CollectLeaves(leaf_vector);
  } else {
    leaf_vector.push_back(this);
  }
}

// Collect pointers to all no-grandchildren nodes.
void StanTree::CollectNogs(std::vector<StanTree*>& nog_vector)
{
  if (left) {
    if (left->left || right->left) {
      if (left->left)  left->CollectNogs(nog_vector);
      if (right->left) right->CollectNogs(nog_vector);
    } else {
      nog_vector.push_back(this);
    }
  }
}

// Collect pointers to all nodes.
void StanTree::CollectNodes(std::vector<StanTree*>& node_vector)
{
  node_vector.push_back(this);
  if (left) {
    left->CollectNodes(node_vector);
    right->CollectNodes(node_vector);
  }
}

void StanTree::CollectNodes(std::vector<const StanTree*>& node_vector) const
{
  node_vector.push_back(this);
  if (left) {
    left->CollectNodes(node_vector);
    right->CollectNodes(node_vector);
  }
}

// Find the leaf reached by observation x using the given cutpoints.
StanTree* StanTree::FindLeaf(double* x, CutpointMatrix& cutpoints)
{
  if (left == nullptr) return this;
  if (std::isnan(x[split_var])) {
    // Missing data: no routing defined
    return nullptr;
  }
  if (x[split_var] < cutpoints[split_var][cut_val]) {
    return left->FindLeaf(x, cutpoints);
  } else {
    return right->FindLeaf(x, cutpoints);
  }
}

// Narrow [lower_bound, upper_bound] for split_var by walking up the ancestors.
void StanTree::FindRegionBounds(size_t split_var, int* lower_bound,
                                int* upper_bound)
{
  if (this->parent == nullptr) return;
  if ((this->parent)->split_var == split_var) {
    if (this == parent->left) {
      if ((int)(parent->cut_val) <= (*upper_bound))
        *upper_bound = (parent->cut_val) - 1;
      parent->FindRegionBounds(split_var, lower_bound, upper_bound);
    } else {
      if ((int)(parent->cut_val) >= *lower_bound)
        *lower_bound = (parent->cut_val) + 1;
      parent->FindRegionBounds(split_var, lower_bound, upper_bound);
    }
  } else {
    parent->FindRegionBounds(split_var, lower_bound, upper_bound);
  }
}

// Reduce the tree to a single root node.
void StanTree::Clear()
{
  size_t tree_size = TreeSize();
  while (tree_size > 1) {
    std::vector<StanTree*> nog_vector;
    CollectNogs(nog_vector);
    for (size_t i = 0; i < nog_vector.size(); i++) {
      delete nog_vector[i]->left;
      delete nog_vector[i]->right;
      nog_vector[i]->left  = nullptr;
      nog_vector[i]->right = nullptr;
    }
    tree_size = TreeSize();
  }
  step_height = 0.0;
  split_var   = 0;
  cut_val     = 0;
  parent = nullptr;
  left   = nullptr;
  right  = nullptr;
}

// Deep-copy the tree rooted at source into destination.
void StanTree::CopyTree(StanTree* destination, const StanTree* source)
{
  if (destination->left) {
    cout << "CopyTree: error, destination node already has children\n";
    return;
  }

  destination->step_height = source->step_height;
  destination->split_var   = source->split_var;
  destination->cut_val     = source->cut_val;

  if (source->left) {
    destination->left = new StanTree;
    (destination->left)->parent = destination;
    CopyTree(destination->left, source->left);

    destination->right = new StanTree;
    (destination->right)->parent = destination;
    CopyTree(destination->right, source->right);
  }
}

// Assignment operator: clear this tree and deep-copy rhs.
StanTree& StanTree::operator=(const StanTree& rhs)
{
  if (&rhs != this) {
    Clear();
    CopyTree(this, &rhs);
  }
  return *this;
}

// Grow a leaf into an internal node via direct pointer.
void StanTree::BirthAtNode(StanTree* leaf, size_t split_var, size_t cut_val,
                           double left_step_height, double right_step_height)
{
  StanTree* left_child  = new StanTree;
  left_child->step_height  = left_step_height;
  StanTree* right_child = new StanTree;
  right_child->step_height = right_step_height;

  leaf->left      = left_child;
  leaf->right     = right_child;
  leaf->split_var = split_var;
  leaf->cut_val   = cut_val;
  left_child->parent  = leaf;
  right_child->parent = leaf;
}

// Remove the children of a nog node via direct pointer.
void StanTree::DeathAtNode(StanTree* nog_node, double new_step_height)
{
  delete nog_node->left;
  delete nog_node->right;
  nog_node->left        = nullptr;
  nog_node->right       = nullptr;
  nog_node->split_var   = 0;
  nog_node->cut_val     = 0;
  nog_node->step_height = new_step_height;
}

// Return the cut value imposed by the nearest ancestor that splits on
// split_var (used for the degenerate-tree augmentation strategy).
size_t StanTree::GetConstrainedCut(size_t split_var)
{
  StanTree* ancestor = this->GetParent();
  if (ancestor->GetSplitVar() == split_var)
    return ancestor->GetCutVal();
  else
    return ancestor->GetConstrainedCut(split_var);
}

