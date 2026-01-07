#include "heat.hpp"
#include "heat_solver.hpp"
#include <fstream>
#include <limits>
#include <iostream>

using namespace mfem;
using namespace std;

double ReadFirstValue(const string &fname, double fallback)
{
  ifstream in(fname);
  if (!in) { cerr << "[q] cannot open " << fname << ", using " << fallback << "\n"; return fallback; }

  double v;
  while (true)
  {
    if (in >> v) return v;                      // got a number
    if (in.eof()) break;                        // nothing left
    in.clear();                                 // token wasn't a number 
    in.ignore(numeric_limits<streamsize>::max(), '\n'); // skip to next line
  }
  cerr << "[q] no numbers found in " << fname << ", using " << fallback << "\n";
  return fallback;
}

ElemValueMap ReadElemValues(const string &path)
{
    ElemValueMap m;

    ifstream in(path);
    if (!in)
    {
        cerr << "ERROR: cannot open file " << path << endl;
        return m;
    }

    long long id;
    double v;

    while (in >> id >> v)
    {
        m[id] = v;   // if duplicate IDs, use the last one 
    }

    // in closes automatically here
    return m;
}

void FillL2FromMap(const Mesh &mesh,
                   const ElemValueMap &id2val,
                   bool ids_one_based,
                   GridFunction &q)
{
    const FiniteElementSpace *fes = q.FESpace();
    Array<int> vdofs;

    // Loop over entries in the map, not over all elements
    for (const auto &p : id2val)
    {
        long long file_id = p.first;
        double value      = p.second;

        int el;
        if (ids_one_based)
        {
            el = static_cast<int>(file_id - 1); // MFEM element index
        }
        else
        {
            el = static_cast<int>(file_id);
        }

        if (el < 0 || el >= mesh.GetNE())
        {
            cerr << "Warning: element id " << file_id
                      << " maps to el=" << el
                      << " which is out of range, skipping.\n";
            continue;
        }

        fes->GetElementVDofs(el, vdofs);   // one DOF for L2^0
        //q(vdofs[0]) = value;
	for (int i = 0; i < vdofs.Size(); i++)
        {
          q(vdofs[i]) = value;
        }
    }
}


