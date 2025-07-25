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

#include "mfem-fe-m_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

using namespace common;

namespace electromagnetics
{

Mfem_fe_mSolver::Mfem_fe_mSolver(ParMesh & pmesh, int order,
                         Coefficient & muInvCoef,
			 Coefficient & sigmaCoef, 
                         const ParGridFunction j_src, 
                         const ParGridFunction phi_src,
			 const int jmode) 
			 //void   (*j_src)(const Vector&, Vector&),
   : myid_(0),
     num_procs_(1),
     order_(order),
     pmesh_(&pmesh),
     visit_dc_(NULL),
     H1FESpace_(NULL),
     HCurlFESpace_(NULL),
     HDivFESpace_(NULL),
     curlMuInvCurl_(NULL),
     hCurlMass_(NULL),
     hDivHCurlMuInv_(NULL),
     weakCurlMuInv_(NULL),
     grad_(NULL),
     curl_(NULL),
     a_(NULL),
     b_(NULL),
     h_(NULL),
     jr_(NULL),
     j_(NULL),
     bd_(NULL),
     jd_(NULL),
     DivFreeProj_(NULL),
     muInvCoef_(&muInvCoef),
     sigmaCoef_(&sigmaCoef),
     aBCCoef_(NULL),
     jCoef_(NULL),
     j_src_(j_src),
     phi_src_(phi_src),
     jmode_(jmode)
{
   // Initialize MPI variables
   MPI_Comm_size(pmesh_->GetComm(), &num_procs_);
   MPI_Comm_rank(pmesh_->GetComm(), &myid_);

   // Define compatible parallel finite element spaces on the parallel
   // mesh. Here we use arbitrary order H1, Nedelec, and Raviart-Thomas finite
   // elements.
   H1FESpace_    = new H1_ParFESpace(pmesh_,order,pmesh_->Dimension());
   HCurlFESpace_ = new ND_ParFESpace(pmesh_,order,pmesh_->Dimension());
   HDivFESpace_  = new RT_ParFESpace(pmesh_,order,pmesh_->Dimension());

   cout<<"size 0 "<<H1FESpace_->TrueVSize()<<endl;
   int irOrder = H1FESpace_->GetElementTransformation(0)->OrderW()
                 + 2 * order;
   int geom = H1FESpace_->GetFE(0)->GetGeomType();
   const IntegrationRule * ir = &IntRules.Get(geom, irOrder);

   // Select surface attributes for Dirichlet BCs on the magnetic field
   // works only for the current mesh
   ess_bdr_.SetSize(pmesh.bdr_attributes.Max());
   ess_bdr_ = 0;   // All outer surfaces
   ess_bdr_[0] = 1;   // symplane coil
   ess_bdr_[1] = 1;   // all coil outer surface
   ess_bdr_[3] = 1;   // symplane air
   ess_bdr_[4] = 1;   // symplance case
   
   non_k_bdr_.SetSize(pmesh.bdr_attributes.Max());
   non_k_bdr_ = 1; // Surfaces without applied surface currents


   // Setup various coefficients

   // Zero Vector Potential on the outer surface
   Vector Zero(3);
   Zero = 0.0;
   aBCCoef_ = new VectorConstantCoefficient(Zero);


   // Construct curl curl solver with stablization scalar delta
   curlMuInvCurl_  = new ParBilinearForm(HCurlFESpace_);
   ConstantCoefficient* delta = new ConstantCoefficient(1e-4);
   curlMuInvCurl_->AddDomainIntegrator(new VectorFEMassIntegrator(*delta));

   curlMuInvCurl_->AddDomainIntegrator(new CurlCurlIntegrator(*muInvCoef_));

   BilinearFormIntegrator * hCurlMassInteg = new VectorFEMassIntegrator;
   hCurlMassInteg->SetIntRule(ir);
   hCurlMass_      = new ParBilinearForm(HCurlFESpace_);
   hCurlMass_->AddDomainIntegrator(hCurlMassInteg);

   BilinearFormIntegrator * hDivHCurlInteg = new VectorFEMassIntegrator(*muInvCoef_);
   hDivHCurlInteg->SetIntRule(ir);
   hDivHCurlMuInv_ = new ParMixedBilinearForm(HDivFESpace_, HCurlFESpace_);
   hDivHCurlMuInv_->AddDomainIntegrator(hDivHCurlInteg);

   // Discrete Curl operator
   curl_ = new ParDiscreteCurlOperator(HCurlFESpace_, HDivFESpace_);

   // Build grid functions
   a_  = new ParGridFunction(HCurlFESpace_);
   b_  = new ParGridFunction(HDivFESpace_);
   h_  = new ParGridFunction(HCurlFESpace_);
   bd_ = new ParGridFunction(HCurlFESpace_);
   jd_ = new ParLinearForm(HCurlFESpace_);

   jr_          = new ParGridFunction(HCurlFESpace_);
   j_           = new ParGridFunction(HCurlFESpace_);
   DivFreeProj_ = new DivergenceFreeProjector(*H1FESpace_, *HCurlFESpace_,
                                                 irOrder, NULL, NULL, grad_);

}

Mfem_fe_mSolver::~Mfem_fe_mSolver()
{
   delete jCoef_;
   delete aBCCoef_;

   delete DivFreeProj_;

   delete a_;
   delete b_;
   delete h_;
   delete jr_;
   delete j_;
   delete bd_;
   delete jd_;

   delete grad_;
   delete curl_;

   delete curlMuInvCurl_;
   delete hCurlMass_;
   delete hDivHCurlMuInv_;
   delete weakCurlMuInv_;

   delete H1FESpace_;
   delete HCurlFESpace_;
   delete HDivFESpace_;

   map<string,socketstream*>::iterator mit;
   for (mit=socks_.begin(); mit!=socks_.end(); mit++)
   {
      delete mit->second;
   }
}

HYPRE_BigInt
Mfem_fe_mSolver::GetProblemSize()
{
   return HCurlFESpace_->GlobalTrueVSize();
}

void
Mfem_fe_mSolver::PrintSizes()
{
   cout<<"dim "<<pmesh_->Dimension()<<endl;
   HYPRE_BigInt size_h1 = H1FESpace_->GlobalTrueVSize();
   HYPRE_BigInt size_nd = HCurlFESpace_->GlobalTrueVSize();
   HYPRE_BigInt size_rt = HDivFESpace_->GlobalTrueVSize();
   if (myid_ == 0)
   {
      cout << "Number of H1      unknowns: " << size_h1 << endl;
      cout << "Number of H(Curl) unknowns: " << size_nd << endl;
      cout << "Number of H(Div)  unknowns: " << size_rt << endl;
   }
}

void
Mfem_fe_mSolver::Assemble()
{
   if (myid_ == 0) { cout << "Assembling ..." << flush; }

   curlMuInvCurl_->Assemble();
   curlMuInvCurl_->Finalize();

   hDivHCurlMuInv_->Assemble();
   hDivHCurlMuInv_->Finalize();

   hCurlMass_->Assemble();
   hCurlMass_->Finalize();

   curl_->Assemble();
   curl_->Finalize();

   if ( grad_ )
   {
      grad_->Assemble();
      grad_->Finalize();
   }
   if ( weakCurlMuInv_ )
   {
      weakCurlMuInv_->Assemble();
      weakCurlMuInv_->Finalize();
   }

   if (myid_ == 0) { cout << " done." << endl; }
}

void
Mfem_fe_mSolver::Update()
{
   if (myid_ == 0) { cout << "Updating ..." << endl; }

   // Inform the spaces that the mesh has changed
   // Note: we don't need to interpolate any GridFunctions on the new mesh
   // so we pass 'false' to skip creation of any transformation matrices.
   H1FESpace_->Update(false);
   HCurlFESpace_->Update(false);
   HDivFESpace_->Update(false);

   HCurlFESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);

   // Inform the grid functions that the space has changed.
   a_->Update();
   h_->Update();
   b_->Update();
   bd_->Update();
   jd_->Update();
   if ( jr_ ) { jr_->Update(); }
   if ( j_  ) {  j_->Update(); }

   // Inform the bilinear forms that the space has changed.
   curlMuInvCurl_->Update();
   hCurlMass_->Update();
   hDivHCurlMuInv_->Update();
   if ( weakCurlMuInv_ ) { weakCurlMuInv_->Update(); }

   // Inform the other objects that the space has changed.
   curl_->Update();
   if ( grad_        ) { grad_->Update(); }
   if ( DivFreeProj_ ) { DivFreeProj_->Update(); }
}

void
Mfem_fe_mSolver::Solve()
{
   if (myid_ == 0) { cout << "Running solver ... " << endl; }

   // Initialize the magnetic vector potential with its boundary conditions
   *a_ = 0.0;

   // Apply uniform B boundary condition on remaining surfaces
   a_->ProjectBdrCoefficientTangent(*aBCCoef_, non_k_bdr_);

   

   /* another potential way of computing j 
   ParMixedBilinearForm grad2(H1FESpace_, HCurlFESpace_);
   grad2.AddDomainIntegrator(new MixedVectorGradientIntegrator());
   grad2.Assemble();
   ParGridFunction E(HCurlFESpace_);
   E = 0.;
   grad2.Mult(phi_src_,E);
   */
   
   if (jmode_ == 1)
   {
     ParDiscreteLinearOperator *grad = new ParDiscreteLinearOperator(H1FESpace_, HCurlFESpace_);
     grad->AddDomainInterpolator(new GradientInterpolator());
     grad->Assemble();

     ParGridFunction E(HCurlFESpace_);
     E = 0.0;
     grad->Mult(phi_src_,E);
     Vector sigmav(pmesh_->attributes.Max());
     sigmav = 0.;
     sigmav(12) = 1.0*2.0*M_PI*10;
     PWConstCoefficient sigmacoef(sigmav);
 
  
     // Initialize the volumetric current density
     *j_ = 0.0;
     ParBilinearForm*  m1 = new ParBilinearForm(HCurlFESpace_);
     m1->AddDomainIntegrator(new VectorFEMassIntegrator(sigmacoef));
     m1->Assemble();
     m1->AddMult(E,*j_,-1.0); // E needs to be HCurl and is grad Phi, use a HCurl field then transfer?   

   }
   else if (jmode_ == 2)
   {
     // Initialize the RHS vector to zero
     *jd_ = 0.0;
     // j_src is  the current density source 
     jCoef_ = new VectorGridFunctionCoefficient(&j_src_);
     jd_->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*jCoef_));
     jd_->Assemble();
   }
   else
   {
     MFEM_VERIFY( jmode_ == 1 || jmode_ == 2, "only two modes of calculating current density"); 
   }
   // Apply Dirichlet BCs to matrix and right hand side and otherwise
   // prepare the linear system
   HypreParMatrix CurlMuInvCurl;
   HypreParVector A(HCurlFESpace_);
   HypreParVector RHS(HCurlFESpace_);
   HCurlFESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);


   if (jmode_ == 1)
   {
     curlMuInvCurl_->FormLinearSystem(ess_bdr_tdofs_, *a_, *j_, CurlMuInvCurl,
                                      A, RHS);
   }
   else
   {
     curlMuInvCurl_->FormLinearSystem(ess_bdr_tdofs_, *a_, *jd_, CurlMuInvCurl,
                                    A, RHS);
   }

   // Define and apply a parallel PCG solver for AX=B with the AMS
   // preconditioner from hypre.
   HypreAMS ams(CurlMuInvCurl, HCurlFESpace_);
   //ams.SetSingularProblem(); // This is a critical option, which is disabled for now. It is used in MFEM Tesla. 

   HyprePCG pcg (CurlMuInvCurl);
   pcg.SetTol(1e-12);
   pcg.SetMaxIter(200);
   pcg.SetPrintLevel(5);
   pcg.SetPreconditioner(ams);
   pcg.Mult(RHS, A);
   
   /*    
   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetPrintLevel(1);
   gmres.SetRelTol(1e-12);
   gmres.SetMaxIter(2000);
   gmres.SetKDim(500);
   gmres.SetOperator(CurlMuInvCurl);
   gmres.SetPreconditioner(ams);
   gmres.Mult(RHS,A);
   */
   // Extract the parallel grid function corresponding to the finite
   // element approximation A. This is the local solution on each
   // processor.

   
   if (jmode_ == 1)
   {
     curlMuInvCurl_->RecoverFEMSolution(A, *j_, *a_);
   }
   else
   {
     curlMuInvCurl_->RecoverFEMSolution(A, *jd_, *a_);
   }
   // Compute the negative Gradient of the solution vector.  This is
   // the magnetic field corresponding to the scalar potential
   // represented by phi.
   curl_->Mult(*a_, *b_);

   // save a field if needed
   /*ofstream a_ofs("a.gf");
   a_ofs.precision(8);
   a_->Save(a_ofs);
   */
   // Compute magnetic field (H) from B and M
   if (myid_ == 0) { cout << "Computing H ... " << flush; }

   hDivHCurlMuInv_->Mult(*b_, *bd_);

   HypreParMatrix MassHCurl;
   Vector BD, H;

   Array<int> dbc_dofs_h;
   hCurlMass_->FormLinearSystem(dbc_dofs_h, *h_, *bd_, MassHCurl, H, BD);

   HyprePCG pcgM(MassHCurl);
   pcgM.SetTol(1e-12);
   pcgM.SetMaxIter(500);
   pcgM.SetPrintLevel(0);
   HypreDiagScale diagM;
   pcgM.SetPreconditioner(diagM);
   pcgM.Mult(BD, H);

   hCurlMass_->RecoverFEMSolution(H, *bd_, *h_);

   if (myid_ == 0) { cout << "done." << flush; 
   
     {
     
     }
   
   }

   if (myid_ == 0) { cout << " Solver done. " << endl; }
}

void
Mfem_fe_mSolver::GetErrorEstimates(Vector & errors)
{
   if (myid_ == 0) { cout << "Estimating Error ... " << flush; }

   //ConstantCoefficient*  coef = new ConstantCoefficient(1.0);
   // Space for the discontinuous (original) flux
   CurlCurlIntegrator flux_integrator(*muInvCoef_);
   RT_FECollection flux_fec(order_-1, pmesh_->SpaceDimension());
   ParFiniteElementSpace flux_fes(pmesh_, &flux_fec);

   // Space for the smoothed (conforming) flux
   int norm_p = 1;
   ND_FECollection smooth_flux_fec(order_, pmesh_->Dimension());
   ParFiniteElementSpace smooth_flux_fes(pmesh_, &smooth_flux_fec);

   L2ZZErrorEstimator(flux_integrator, *a_,
                      smooth_flux_fes, flux_fes, errors, norm_p);

   if (myid_ == 0) { cout << "done." << endl; }
}

void
Mfem_fe_mSolver::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   visit_dc_ = &visit_dc;

   visit_dc.RegisterField("A", a_);
   visit_dc.RegisterField("B", b_);
   visit_dc.RegisterField("H", h_);
   if ( j_ ) { visit_dc.RegisterField("J", j_); }
}

void
Mfem_fe_mSolver::WriteVisItFields(int it)
{
   if ( visit_dc_ )
   {
      if (myid_ == 0) { cout << "Writing VisIt files ..." << flush; }

      HYPRE_BigInt prob_size = this->GetProblemSize();
      visit_dc_->SetCycle(it);
      visit_dc_->SetTime(prob_size);
      visit_dc_->Save();

      if (myid_ == 0) { cout << " done." << endl; }
   }
}

void
Mfem_fe_mSolver::InitializeGLVis()
{
   if ( myid_ == 0 ) { cout << "Opening GLVis sockets." << endl; }

   socks_["A"] = new socketstream;
   socks_["A"]->precision(8);

   socks_["B"] = new socketstream;
   socks_["B"]->precision(8);

   socks_["H"] = new socketstream;
   socks_["H"]->precision(8);

   if ( j_ )
   {
      socks_["J"] = new socketstream;
      socks_["J"]->precision(8);
   }
   if ( myid_ == 0 ) { cout << "GLVis sockets open." << endl; }
}

void
Mfem_fe_mSolver::DisplayToGLVis()
{
   if (myid_ == 0) { cout << "Sending data to GLVis ..." << flush; }

   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 350, Wh = 350; // window size
   int offx = Ww+10, offy = Wh+45; // window offsets

   VisualizeField(*socks_["A"], vishost, visport,
                  *a_, "Vector Potential (A)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["B"], vishost, visport,
                  *b_, "Magnetic Flux Density (B)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["H"], vishost, visport,
                  *h_, "Magnetic Field (H)", Wx, Wy, Ww, Wh);
   Wx += offx;

   if ( j_ )
   {
      VisualizeField(*socks_["J"], vishost, visport,
                     *j_, "Current Density (J)", Wx, Wy, Ww, Wh);
   }

   Wx = 0; Wy += offy; // next line

   if (myid_ == 0) { cout << " done." << endl; }
}
} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI
