#ifndef GUARD_StanBirthDeath_h
#define GUARD_StanBirthDeath_h

#include "Info.h"
#include "StanTree.h"
#include "StanTreeFunctions.h"
#include "StanForestFunctions.h"

// Propose and accept/reject a single birth or death step for tree using a
// Metropolis-Hastings ratio.  Returns true if the proposal was accepted.
bool BirthDeathStep(StanTree& tree, CutpointMatrix& cutpoints,
                    DataInfo& data_info, PriorInfo& prior_info,
                    double sigma,
                    std::vector<size_t>& variable_split_counts,
                    std::vector<double>& split_probabilities,
                    bool use_augmentation, Random& random);

// IRS-aware overload: passes routing map for missing data handling.
// irs_mode: 1=skip-then-draw, 2=draw-then-decide, 3=uniform (P=0.5).
bool BirthDeathStep(StanTree& tree, CutpointMatrix& cutpoints,
                    DataInfo& data_info, PriorInfo& prior_info,
                    double sigma,
                    std::vector<size_t>& variable_split_counts,
                    std::vector<double>& split_probabilities,
                    bool use_augmentation, Random& random,
                    RoutingMap& routing_map, int irs_mode = 1);

#endif

