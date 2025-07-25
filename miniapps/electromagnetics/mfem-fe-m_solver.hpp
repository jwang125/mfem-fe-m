// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TESLA_SOLVER
#define MFEM_TESLA_SOLVER

#include "../common/pfem_extras.hpp"
#include "../common/mesh_extras.hpp"
#include "electromagnetics.hpp"

#ifdef MFEM_USE_MPI

#include <string>
#include <map>

namespace mfem
{

using common::H1_ParFESpace;
using common::ND_ParFESpace;
using common::RT_ParFESpace;
using common::ParDiscreteGradOperator;
using common::ParDiscreteCurlOperator;
using common::DivergenceFreeProjector;

namespace electromagnetics
{

class Mfem_fe_mSolver
{
public:
   Mfem_fe_mSolver(ParMesh & pmesh, int order, 
               Coefficient & muInvCoef, Coefficient & sigmaCoef, 
	       const ParGridFunction j_src,
               const ParGridFunction phi_src, const int jmode);
               //void   (*j_src)(const Vector&, Vector&),
   ~Mfem_fe_mSolver();

   HYPRE_BigInt GetProblemSize();

   void PrintSizes();

   void Assemble();

   void Update();

   void Solve();

   void GetErrorEstimates(Vector & errors);

   void RegisterVisItFields(VisItDataCollection & visit_dc);

   void WriteVisItFields(int it = 0);

   void InitializeGLVis();

   void DisplayToGLVis();

   const ParGridFunction & GetVectorPotential() { return *a_; }

private:

   int myid_;
   int num_procs_;
   int order_;
   int jmode_;

   ParMesh * pmesh_;

   VisItDataCollection * visit_dc_;

   H1_ParFESpace * H1FESpace_;
   ND_ParFESpace * HCurlFESpace_;
   RT_ParFESpace * HDivFESpace_;

   ParBilinearForm * curlMuInvCurl_;
   ParBilinearForm * hCurlMass_;
   ParMixedBilinearForm * hDivHCurlMuInv_;
   ParMixedBilinearForm * weakCurlMuInv_;

   ParDiscreteGradOperator * grad_;
   ParDiscreteCurlOperator * curl_;

   ParGridFunction * a_;  // Vector Potential (HCurl)
   ParGridFunction * b_;  // Magnetic Flux (HDiv)
   ParGridFunction * h_;  // Magnetic Field (HCurl)
   ParGridFunction * jr_; // Raw Volumetric Current Density (HCurl)
   ParGridFunction * j_;  // Volumetric Current Density (HCurl)
   ParGridFunction * bd_; // Dual of B (HCurl)
   ParLinearForm * jd_; // Dual of J, the rhs vector (HCurl)

   DivergenceFreeProjector * DivFreeProj_;

   Coefficient       * muInvCoef_; // Dia/Paramagnetic Material Coefficient
   Coefficient       * sigmaCoef_; // Conductivity Material Coefficient
   VectorCoefficient * aBCCoef_;   // Vector Potential BC Function
   VectorCoefficient * jCoef_;     // Volume Current Density Function
  
   ParGridFunction j_src_;
   
   ParGridFunction phi_src_;

   //void   (*j_src_)(const Vector&, Vector&);

   Array<int> ess_bdr_;
   Array<int> ess_bdr_tdofs_;
   Array<int> non_k_bdr_;

   std::map<std::string,socketstream*> socks_;
};

} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_TESLA_SOLVER
