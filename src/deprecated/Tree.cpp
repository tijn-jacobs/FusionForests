#include "Tree.h"

// Function returns the identification number of a node. Numbering is based on
// a "full" tree.
size_t Tree::NodeID() const {
  
  // Check if the node is the root (stump).
  if (parent == nullptr) {
    
    return 1;
  }
  
  // Recursively determine the identification number.
  if (this == parent->left) {
    
    // If the node is a left child.
    return 2 * (parent->NodeID());
    
  } else {
    
    // If the node is a right child.
    return 2 * (parent->NodeID()) + 1;
  } 
} // Updated


// Function to get a pointer to the node with the given identification number.
Tree* Tree::GetNodePointer(size_t ID) {
  
  // Check if the current node has the desired ID.
  if (this->NodeID() == ID) {
    
    return this; // The current node is the correct node.
  }
  
  // If the current node has no left child, it is a leaf i.e. bottom node. It
  // does not have any descendants, so return nullptr. This is actually 
  // superfluous, but increases performance.
  if (left == nullptr) {
    return nullptr; // No children, node not found.
  } 
  
  // Recursively search for the node in the left subtree.
  Tree* left_child = left->GetNodePointer(ID);
  if (left_child != nullptr) {
    return left_child; // Found the node in the left subtree.
  } 
  
  // Recursively search for the node in the right subtree.
  Tree* right_child = right->GetNodePointer(ID);
  if (right_child != nullptr) {
    return right_child; // Found the node in the right subtree.
  } 
  
  return nullptr; // Node not found in the tree.
} // Updated


// Function to return the depth of the current node. Depth equals the amount of
// ancestoral generations.
size_t Tree::NodeDepth() const {
  
  // If the node has no parent, it is the root, so depth is 0.
  if (parent == nullptr) {
    
    return 0;
    
  } else {
    
    // Otherwise, the depth is one more than the depth of the parent.
    return 1 + parent->NodeDepth();
    
  }
} // Updated


// Function to return the total number of nodes in the tree. Traverses down the 
// tree and adds 1 to the total for each node is encounters.
size_t Tree::TreeSize() const {
  
  // If the node is terminal, the size of the subtree is 1.
  if (left == nullptr) {
    
    // Add 1 to the total size of the tree.
    return 1;
  } else {
    
    // Otherwise, size is 1 (current node) plus sizes of left and right subtrees.
    return 1 + left->TreeSize() + right->TreeSize();
  }
} // Updated


// Function to print out the tree or node information.
// If also_print_children is true, prints the entire tree; if false, prints the 
// current node.
void Tree::PrintTree(bool also_print_children) const {
  size_t node_depth = NodeDepth();
  size_t ID = NodeID();
  size_t parent_id = (parent == nullptr) ? 0 : parent->NodeID();
  
  // Define auxilliary printing variables.
  std::string padding(2 * node_depth, ' ');
  std::string separator(", ");
  
  // Print tree size if printing the whole tree and this is the top node.
  if (also_print_children && (parent == nullptr)) {
    cout << "Tree size: " << TreeSize() << std::endl;
  } 
  
  // Print node details.
  cout << padding << "(ID, parent): " << ID << separator << parent_id
            << separator << "(v, c): " << split_var << separator << cut_val
            << separator << "Parameter: " << GetParameters(0)
            << separator << "Depth of the node: " << node_depth
            << separator << "Node address: " << this << std::endl;
  
  // Recursively print children if also_print_children is true.
  if (also_print_children) {
    if (left != nullptr) {
      left->PrintTree(also_print_children);
    } 
    if (right != nullptr) {
      right->PrintTree(also_print_children);
    }
  }
}  // Updated


// Function to check if the node is a nog node (a node with no grandchildren).
// This is equivalent to having exactly two descendants.
bool Tree::IsNog() const {
  
  // Check if the node has children. Otherwise, not a nog.
  if (left == nullptr) {
    return false;
  }
  
  // Check if either child has children. In that case, not a nog.
  if (left->left != nullptr || right->left != nullptr) {
    return false; // One of the children has children.
  }
  
  // The node has children who do not have children themselves. Thus a nog.
  return true;
} // Updated


// Function computes the amount of nog nodes in the tree.
size_t Tree::NumberOfNogs() const {
  
  // If the node is a leaf it is not a nog. 
  if (left == nullptr) {
    return 0;
  }
  
  // Recursively check if the nodes in the subtree are a nog node.
  if (left->left != nullptr || right->left != nullptr) {
    return left->NumberOfNogs() + right->NumberOfNogs();
  } else { 
    // The node is a nog node. So return 1.
    return 1;
  } 
} 

// Function to count the number of leaf nodes in the subtree, i.e. the number
// of terminal nodes.
size_t Tree::NumberOfLeaves() const {
  
  // If the node is a leaf (no left child), return 1.
  if (left == nullptr) {
    return 1;
  }
  
  // Recursively count the bottom nodes in the left and right subtrees.
  return left->NumberOfLeaves() + right->NumberOfLeaves();
  
}


// Function retrieves the addresses (i.e. pointers to) of  leaf nodes in the 
// tree.
void Tree::CollectLeaves(std::vector<Tree*>& leaf_vector) {
  
  // If the node has children, recursively retrieve the leaf addresses from 
  // both children.
  if (left != nullptr) {
    
    left->CollectLeaves(leaf_vector);
    right->CollectLeaves(leaf_vector);
    
  } else {
    
    // If the node is a leaf node, add its address to the vector.
    leaf_vector.push_back(this);
  }
}

// Function to collect the addresses of all nog nodes in the tree.
// A "nog" node is defined as a node with no grandchildren.
void Tree::CollectNogs(std::vector<Tree*>& nog_vector) {
  
  // Check whether the current node has any children at all.
  if (left != nullptr) {
    
    // Check whether the current node has grandchildren.
    if (left->left != nullptr || right->left != nullptr) {
      
      // If the node has grandchildren, recursively collect the nogs addresses
      // from both subtrees. 
      left->CollectNogs(nog_vector);
      right->CollectNogs(nog_vector);
      
    } else {
      
      // If the current node has children (but no grandchildren), add it to
      // the vector of vector of nog nodes.
      nog_vector.push_back(this);
    }
  } 
}


// Function to collect all nodes in the tree.
// The nodes are added to the provided vector `node_vector`.
void Tree::CollectNodes(std::vector<Tree*>& node_vector) {
  
  // Add the current node to the vector.
  node_vector.push_back(this);
  
  // Check if the current node is terminal.
  if (left != nullptr) {
    
    // Recursively collect nodes from both children.
    left->CollectNodes(node_vector);
    right->CollectNodes(node_vector);
  }
} 


// Overloaded to collect all nodes in the tree (const version).
// The nodes are added to the provided vector `node_vector`.
void Tree::CollectNodes(std::vector<const Tree*>& node_vector) const {
  
  // Add the current node to the vector.
  node_vector.push_back(this);
  
  // Check if the node is terminal.
  if (left != nullptr) {
    
    // Recursively collect nodes from both children.
    left->CollectNodes(node_vector);
    right->CollectNodes(node_vector);
  }
} 

// Function searches the interval (lower_bound, upper_bound) in which specified
// variable split_var can be split. This function recursively adjusts the  
// interval by traversing up the tree.
void Tree::PossibleCuts(size_t v, int* lower_bound, int* upper_bound) const {
  
  // If the node has no parent, return as we've reached the top of the tree.
  if (parent == nullptr) {
    return;
  }
  
  // Check if the parent node uses the same split variable `v`.
  if (parent->split_var == v) {
    
    // Determine if the current node is the left or right child.
    if (this == parent->left) {
      
      // If the node is the left child, update the upper bound `U`.
      if (static_cast<int>(parent->cut_val) <= *upper_bound) {
        *upper_bound = static_cast<int>(parent->cut_val) - 1;
      }
    } else {
      
      // If the node is the right child, update the lower bound.
      if (static_cast<int>(parent->cut_val) >= *lower_bound) {
        *lower_bound = static_cast<int>(parent->cut_val) + 1;
      }
    }
    
    // Recursively call PossibleCuts on the parent node.
    parent->PossibleCuts(v, lower_bound, upper_bound);
  } else {
    
    // If the parent node does not use the same split variable, continue 
    // the recursion.
    parent->PossibleCuts(v, lower_bound, upper_bound);
  }
}


// Function to cut down the tree to a single node (the root / stump).
void Tree::CutDownTree() {
  
  // Get the current size of the tree.
  size_t tree_size = TreeSize();
  
  // Loop until the tree size is reduced to 1 (only the root node remains).
  // Loop invariant: tree_size >= 1
  while (tree_size > 1) {
    std::vector<Tree*> nog_nodes;
    
    // Collect all "nog" nodes (nodes with no grandchildren).
    CollectNogs(nog_nodes);
    
    // Delete the children of each "nog" node.
    for (size_t i = 0; i < nog_nodes.size(); ++i) {
      delete nog_nodes[i]->left;
      delete nog_nodes[i]->right;
      nog_nodes[i]->left = nullptr;
      nog_nodes[i]->right = nullptr;
    }
    
    // Update the tree size.
    tree_size = TreeSize();
  } 
  
  // Reset the attributes of the root node.
  parameters.Reset();
  split_var = 0;
  cut_val = 0;
  parent = nullptr;
  left = nullptr;
  right = nullptr;
  leaf_index_ = std::numeric_limits<size_t>::max();
} 


// Function to copy the structure and data from the original tree to the new 
// tree. If the new tree has children, it will first delete them. 
void Tree::CopyTree(Tree* new_tree, const Tree* original_tree) {
  
  // If the new tree already has children, kill them.
  if (new_tree->left != nullptr) {
    new_tree->CutDownTree();
  }
  
  // Copy the data from the original tree to the new tree.
  new_tree->parameters = original_tree->parameters;
  new_tree->split_var = original_tree->split_var;
  new_tree->cut_val = original_tree->cut_val;
  
  // If the original tree has children, recursively copy them.
  if (original_tree->left != nullptr) {
    
    // Copy the left child.
    new_tree->left = new Tree;
    new_tree->left->parent = new_tree;
    CopyTree(new_tree->left, original_tree->left);
    
    // Copy the right child.
    new_tree->right = new Tree;
    new_tree->right->parent = new_tree;
    CopyTree(new_tree->right, original_tree->right);
  } 
}


// Function to add children to the terminal node 'leaf'.
// `v` is the split variable, `c` is the cut value, `theta_left` and 
// `theta_right` are the theta values for the left and right children.
void Tree::GrowChildren(Tree* leaf, size_t _split_var, size_t _cut_val, 
                        Parameters par_left, Parameters par_right) {
  
  // Create new left and right child nodes.
  Tree* left = new Tree;
  Tree* right = new Tree;
  
  // Set the theta values for the left and right children.
  left->parameters = par_left;
  right->parameters = par_right;
  
  // Assign the new children to the 'leaf' node.
  leaf->left = left;
  leaf->right = right;
  
  // Set the split variable and cut value for 'leaf' node.
  leaf->split_var = _split_var;
  leaf->cut_val = _cut_val;
  
  // Set the parent pointers for the new children.
  left->parent = leaf;
  right->parent = leaf;

  // Initialize the leaf_index_ for the new children.
  left->leaf_index_  = std::numeric_limits<size_t>::max();
  right->leaf_index_ = std::numeric_limits<size_t>::max();

  // This node (the passed-in leaf) becomes internal: “unset” its index too
  leaf->leaf_index_  = std::numeric_limits<size_t>::max();  // <- was: leaf_index_

}


// Function to remove (delete) the children of a nog node.
// The `parameters` object of the node `nog_node` is updated.
void Tree::KillChildren(Tree* nog_node, Parameters parameters) {
  // Delete left and right children
  delete nog_node->left;
  delete nog_node->right;
  
  // Reset left and right pointers to nullptr
  nog_node->left = nullptr;
  nog_node->right = nullptr;
  
  // Reset the split variable and cut value of the nog node
  nog_node->split_var = 0;
  nog_node->cut_val = 0;
  
  // Set the parameter value of the nog node to the provided `parameters`
  nog_node->parameters = parameters;

  // This node is now a leaf in the current structure
  nog_node->leaf_index_ = std::numeric_limits<size_t>::max();
}

// Overloaded function to remove (delete) the children of a nog node.
// This version doesn't take `parameters` as an argument.
void Tree::KillChildren(Tree* nog_node) {
  // Delete left and right children
  delete nog_node->left;
  delete nog_node->right;
  
  // Reset left and right pointers to nullptr
  nog_node->left = nullptr;
  nog_node->right = nullptr;
  
  // Reset the split variable and cut value of the nog node
  nog_node->split_var = 0;
  nog_node->cut_val = 0;

  // Parameters unchanged; just mark as leaf for the stats pass.
  nog_node->leaf_index_ = std::numeric_limits<size_t>::max();
}


// Function to remove the children of a nog node with the given ID. 
bool Tree::KillChildren(size_t node_ID, Parameters parameters) {
  
  // Get the pointer to the node with the given ID.
  Tree* node = GetNodePointer(node_ID);
  
  // If the node pointer is null, the node ID is invalid.
  if (node == nullptr) {
    cout << "error in KillChildren: invalid node ID\n";
    return false;
  }
  
  // Check if the node is a "nog" node (a node with no grandchildren).
  if (node->IsNog()) {
    // Delete the left and right children.
    delete node->left;
    delete node->right;
    
    // Set the left and right pointers to nullptr.
    node->left = nullptr;
    node->right = nullptr;
    
    // Reset the splitting variable and cutpoint.
    node->split_var = 0;
    node->cut_val = 0;
    
    // Set the node's theta value to the provided theta.
    node->parameters = parameters;

    node->leaf_index_ = std::numeric_limits<size_t>::max();

    return true;
  } else { 
    // If the node is not a "nog" node, print an error message.
    cout << "error in death: node is not a nog node\n";
    return false;
  }
} // Partially updated


// Recursive function to find the cut value for the given `split_var`.
// If the parent node uses the same split variable, return its cut value.
// Otherwise, continue the search up the tree.
/*
size_t Tree::FindSameCut(size_t split_var) const {
  // Get the parent node of the current node
  Tree* parent_node = this->GetParent();
  
  // If parent uses the same split variable, return its cut value
  if (parent_node->GetSplitVar() == split_var) {
    return parent_node->GetCutVal();
  }
  
  // Otherwise, recursively search up the tree
  return parent_node->FindSameCut(split_var);
}
*/
size_t Tree::FindSameCut(size_t split_var) const {
    // Get the parent node of the current node
  Tree* parent_node = this->GetParent();

  if (!parent_node) {
    return 0; // or some safe sentinel; this path is rarely used but must be safe
  }

  // If parent uses the same split variable, return its cut value
  if (parent_node->GetSplitVar() == split_var) {
    return parent_node->GetCutVal();
  }

  // Otherwise, recursively search up the tree
  return parent_node->FindSameCut(split_var);
}



// Evaluate a tree on a given set of covariates
// `estimated_outcomes` contains the estimated outcomes for all observations 
// according to the tree.
void Tree::EvaluateTree(Cutpoints& cutpoints, size_t p, size_t n, 
                        double* x, Data& data, double* estimated_outcomes) {
  // Iterate over each observation
  for (size_t i = 0; i < n; ++i) {
    // Find the leaf node corresponding to the current observation
    Tree* leaf_node = FindLeaf(x + i * p, cutpoints);
    
    // Store the parameter (e.g., mean outcome) from the leaf node as the 
    // estimated outcome for this observation
    estimated_outcomes[i] = leaf_node->GetParameters(0);
  }
}

// Method to print the structure of the tree
void Tree::PrintFullTree(int depth, Cutpoints& cutpoints, Data& data) {
  // Indentation based on depth to visualize the tree structure
  for (int i = 0; i < depth; ++i) {
    cout << "  ";  // Two spaces per depth level
  }

  // Print node ID
  cout << "Node ID: " << NodeID() << ": ";

  // Print split information if it's not a leaf node
  if (left != nullptr || right != nullptr) {
    cout << " (Splitting variable: " << split_var 
              << ", Cut value: " 
              << cutpoints.values[split_var][cut_val] << ")";
  } else {
    // Print parameter and node size for leaf nodes
    cout << " (Parameters: " << GetParameters(0) << ")";
  }

  cout << "\n";

  // Recursively print left and right children
  if (left != nullptr) {
    left->PrintFullTree(depth + 1, cutpoints, data);
  }

  if (right != nullptr) {
    right->PrintFullTree(depth + 1, cutpoints, data);
  }
}

// Method to calculate the size of the node (how many data points fall into it)
size_t Tree::NodeSize(Data& data, Cutpoints& cutpoints) {
  size_t count = 0;

  // Traverse through all observations in the dataset
  for (size_t i = 0; i < data.GetN(); ++i) {
    double* x = data.GetDataRow(i); // Get the i-th observation

    // Use FindLeaf to determine which leaf the observation falls into
    Tree* leaf = FindLeaf(x, cutpoints);

    // Debug output to show which observation falls into which leaf node
    cout << "i: " << i << ", loc: " << leaf << ", x: " << x << endl;

    // Check if this node is the leaf node
    if (leaf->NodeID() == this->NodeID()) {
      ++count;  // Increment count if observation falls into this leaf
    }
  }

  return count;  // Return the total number of observations in this node
}

