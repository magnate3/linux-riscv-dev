/*
no gpu was used. if it was, itll be need 4
*/
#include <iostream>

// This function determines if a given rank is the root of its group.
// It divides nRanks among nIds groups. The groups may have different sizes
// if nRanks isn't exactly divisible by nIds.
bool rankHasRoot(const int rank, const int nRanks, const int nIds) {
  // rmr: remainder of total ranks divided by the number of IDs.
  const int rmr = nRanks % nIds;
  // rpr: quotient, representing the base number of ranks per ID.
  const int rpr = nRanks / nIds;
  // rlim: threshold rank that separates groups with one extra rank.
  const int rlim = rmr * (rpr + 1);

  // For the first rmr groups, each group has (rpr+1) ranks.
  if (rank < rlim) {
    // The first rank in the group (i.e., a "root") will be at an index that's a multiple of (rpr+1)
    return (rank % (rpr + 1)) == 0;
  } else {
    // For the remaining groups, each group has rpr ranks.
    return ((rank - rlim) % rpr) == 0;
  }
}

int main() {
  // Example configuration:
  // Total number of ranks and the number of groups (IDs)
  const int nRanks = 16;
  const int nIds = 4;

  std::cout << "Total Ranks: " << nRanks << ", Number of IDs (groups): " << nIds << "\n";
  std::cout << "Determining which rank is the group root:\n";

  // Loop through each rank and use the function to check if it's a root.
  for (int rank = 0; rank < nRanks; ++rank) {
    bool hasRoot = rankHasRoot(rank, nRanks, nIds);
    std::cout << "Rank " << rank << (hasRoot ? " is a root." : " is NOT a root.") << "\n";
  }

  return 0;
}
