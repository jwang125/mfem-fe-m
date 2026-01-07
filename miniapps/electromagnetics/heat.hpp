#pragma once
#include "mfem.hpp"
#include <string>

using namespace std;
using namespace mfem;

using ElemValueMap = unordered_map<long long, double>;

// Read the first numeric value from a text file (ignores blank lines and lines starting with '#').
// Falls back to 0.0 if the file is missing or has no numbers.
// Not in use.
double ReadFirstValue(const string &fname, double fallback = 0.0);


// read "<elem_id> <value>" text file into a map 
ElemValueMap ReadElemValues(const string &path);

// fill an L2 GridFunction with per-element values
// The map id2val contains: element_id to value.
// If the text file uses 1-based IDs (Exodus style), set ids_one_based = true.
// currently supports pmesh for parallel code
void FillL2FromMap(const Mesh &mesh,
                   const ElemValueMap &id2val,
                   bool ids_one_based,
                   GridFunction &q);

