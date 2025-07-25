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
//
//            -----------------------------------------------------
//            MFEM-FE-M Miniapp:  Fusion energy - Magnets simulation
//            -----------------------------------------------------
//
// This miniapp solves a model fusion energy problem.
//
//
//
//
//
// Compile with: make mfem-fe-m
//
//

#include "mfem-fe-m_solver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::electromagnetics;

// Permeability Function
Coefficient * SetupInvPermeabilityCoefficient();

static Vector pw_mu_(0);      // Piecewise permeability values
static Vector pw_mu_inv_(0);  // Piecewise inverse permeability values



// Target integrated current level   
real_t target_current = -1.0;
// Prints the program's logo to the given output stream
void display_banner(ostream & os);
void ComputeCurrentDensityOnSubMesh(int order,
                                    bool visualization,
				    Coefficient &sigmaCoef, 
                                    const Array<int> &phi0_attr,
                                    const Array<int> &phi1_attr,
                                    const Array<int> &jn_zero_attr,
                                    ParGridFunction &phi_h1,
                                    ParGridFunction &j_cond,
                                    ParGridFunction &j_l2);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   if ( Mpi::Root() ) { display_banner(cout); }

   // Parse command-line options.
   const char *mesh_file = "halfwedge2.g";
   int order = 2;
   // default iteration is 1
   int maxit = 1;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   // default visulation set to true
   bool visualization = true;
   bool visit = true;
   
   // two methods of integrating current density are implemented
   // option 1 (default) uses the joule approach that calculates a grad Phi in HCurl 
   // option 2 uses l2 current density and VectorGridFunctionCoefficient  
   int jmode = 1;

   // only cpu has been tested
   const char *device_config = "cpu";


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&pw_mu_, "-pwm", "--piecewise-mu",
                  "Piecewise values of Permeability");
   args.AddOption(&maxit, "-maxit", "--max-amr-iterations",
                  "Max number of iterations in the main AMR loop.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&jmode, "-jm", "--jmode",
                  "Methods to compute current density. See tesla_solver.cpp for description.");
   args.AddOption(&target_current, "-tc", "--target_current",
                  "Adjust current density if there is a target current.");
   args.Parse();
 

   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }
   // Read the (serial) mesh from the given mesh file on all processors.  
   // Only one mesh is supported for now. 
   Mesh *mesh = new Mesh(mesh_file, 1, 1);

   if (Mpi::Root())
   {
      cout << "Starting initialization." << endl;
   }


   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement.
   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }
  
   // the boundary condition is specified for the wedge mesh 
   Array<int> cond_attr(1);
   // for the half wedge
   cond_attr[0] = 13;
   

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   cout<<"creating submesh "<<endl;
   ParSubMesh mesh_cond(ParSubMesh::CreateFromDomain(pmesh, cond_attr));

   // Refine this mesh in parallel to increase the resolution.
   int par_ref_levels = parallel_ref_levels;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }
   // Make sure tet-only meshes are marked for local refinement.
   //pmesh.Finalize(true);

   // ************************************************
   // Solve for the J field
   // Extract a submesh covering the coil portion of the domain
     

   // Define a suitable finite element space on the SubMesh and compute
   //    the current density as an H(div) field.
   RT_FECollection fec_cond_rt(order-1, pmesh.Dimension());
   ParFiniteElementSpace fes_cond_rt(&mesh_cond, &fec_cond_rt);
   ParGridFunction j_cond(&fes_cond_rt);

   // Now define a L2 current density function
   L2_FECollection fec_cond_l2(order-1, pmesh.Dimension());
   ParFiniteElementSpace fes_cond_l2(&mesh_cond, &fec_cond_l2, pmesh.Dimension());
   ParGridFunction j_l2(&fes_cond_l2);

   Array<int> phi0_attr;
   Array<int> phi1_attr;
   Array<int> jn_zero_attr;
   // for the half wedge
   phi0_attr.Append(1);
   //phi1_attr.Append(2);
   phi1_attr.Append(2);
   
   Vector sigmav(mesh_cond.attributes.Max());
   sigmav = 0.;
   //sigmav = 1e-6*2.0*M_PI*10;
   sigmav(12) = 1.0*2.0*M_PI*10;
   PWConstCoefficient sigmaCoef(sigmav);
   

   H1_FECollection fec_h1_cond(order, pmesh.Dimension());
   ParFiniteElementSpace fes_h1_cond(&mesh_cond, &fec_h1_cond);
   ParGridFunction phi_cond(&fes_h1_cond);
   phi_cond = 0.0;

   // Both the H(div) and L2 j field are solved
   ComputeCurrentDensityOnSubMesh(order, visualization, sigmaCoef, 
                                  phi0_attr, phi1_attr, jn_zero_attr, phi_cond, j_cond, j_l2);
   // Solve J field end

   // Transfer to full mesh: H1 phi, Hdiv J, and L2 J
   H1_FECollection fec_h1(order, pmesh.Dimension());
   ParFiniteElementSpace fespace_h1(&pmesh, &fec_h1);
   ParGridFunction phi_full(&fespace_h1);
   phi_full = 0.0;
   mesh_cond.Transfer(phi_cond, phi_full);

   RT_FECollection fec_rt(order-1, pmesh.Dimension());
   ParFiniteElementSpace fespace_rt(&pmesh, &fec_rt);
   ParGridFunction j_full(&fespace_rt);
   j_full = 0.0;
   mesh_cond.Transfer(j_cond, j_full);

   L2_FECollection fec_l2(order-1, pmesh.Dimension());
   ParFiniteElementSpace fes_l2(&pmesh, &fec_l2, pmesh.Dimension());
   ParGridFunction j_full_l2(&fes_l2);
   j_full_l2 = 0.0;
   mesh_cond.Transfer(j_l2, j_full_l2);


   /*
   {  store j is needed
      ofstream j_of("j.gf");
      j_of.precision(8);
      j_full.Save(j_of);
   }
   */

   // Integrate the current on the boundary surface  
   Array<int> neumann_bdr;

   Array<int> ess_bdr_phi(pmesh.bdr_attributes.Max());
   ess_bdr_phi = 0;
   //ess_bdr_phi[0] = 1;
   ess_bdr_phi[1] = 1;
   fespace_rt.GetEssentialTrueDofs(ess_bdr_phi, neumann_bdr); 
   
   ParGridFunction j_one(&fespace_rt);
   double j_integrate=0;
   ParLinearForm* temp_j  = new ParLinearForm(&fespace_rt);
   VectorGridFunctionCoefficient jbnd(&j_full);
   temp_j->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(jbnd),ess_bdr_phi);
   temp_j->Assemble();
   j_one = 1.0; 
   j_integrate = (*temp_j)(j_one); // Linear form apply?
   cout<<"integrated current"<<j_integrate<<endl;

   // Adjust the conductivity to scale the current density to target value
   if(target_current > 0)
   {
     real_t current_ratio = target_current/abs(j_integrate);
     
     Vector sigmav_scaled(mesh_cond.attributes.Max());
     sigmav_scaled = 0.;
     sigmav_scaled(12) = 1.0*2.0*M_PI*10*current_ratio;
     cout<<"sigma v scaled "<<endl;
     sigmav_scaled.Print();
     PWConstCoefficient sigmaCoef_scaled(sigmav_scaled);
 
     ComputeCurrentDensityOnSubMesh(order, visualization, sigmaCoef_scaled, 
                                  phi0_attr, phi1_attr, jn_zero_attr, phi_cond, j_cond, j_l2);
     mesh_cond.Transfer(j_cond, j_full);
     mesh_cond.Transfer(j_l2, j_full_l2);
     //how to update VectorGridFunctionCoefficient?
     double j_integrate_scaled=0;
     ParLinearForm* temp_j_scaled  = new ParLinearForm(&fespace_rt);
     VectorGridFunctionCoefficient jbnd_scaled(&j_full);
     temp_j_scaled->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(jbnd_scaled),ess_bdr_phi);
     temp_j_scaled->Assemble();
     j_integrate_scaled = (*temp_j_scaled)(j_one); // Linear form apply?
     cout<<"integrated current scaled"<<j_integrate_scaled<<endl;
   }


   int myid_;      // Local processor rank
   MPI_Comm_rank(pmesh.GetComm(), &myid_);

   // Create a coefficient describing the magnetic permeability
   Coefficient * muInvCoef = SetupInvPermeabilityCoefficient();

   // Create the Magnetostatic solver
   Mfem_fe_mSolver Mfem_fe_m(pmesh, order, *muInvCoef, sigmaCoef, j_full_l2, phi_full, jmode);

   // Initialize GLVis visualization
   if (visualization)
   {
      Mfem_fe_m.InitializeGLVis();
   }

   // Initialize VisIt visualization
   VisItDataCollection visit_dc("Mfem_fe_m-Parallel", &pmesh);

   if ( visit )
   {
      Mfem_fe_m.RegisterVisItFields(visit_dc);
      visit_dc.RegisterField("j_hdiv", &j_full);
      visit_dc.RegisterField("j_l2", &j_full_l2);
   }
   if (Mpi::Root()) { cout << "Initialization done." << endl; }

   // The main AMR loop. In each iteration we solve the problem on the current
   // mesh, visualize the solution, estimate the error on all elements, refine
   // the worst elements and update all objects to work with the new mesh. We
   // refine until the maximum number of dofs in the Nedelec finite element
   // space reaches 10 million.
   const int max_dofs = 10000000;
   for (int it = 1; it <= maxit; it++)
   {
      if (Mpi::Root())
      {
         cout << "\nAMR Iteration " << it << endl;
      }

      // Display the current number of DoFs in each finite element space
      Mfem_fe_m.PrintSizes();

      // Assemble all forms
      Mfem_fe_m.Assemble();

      // Solve the system and compute any auxiliary fields
      Mfem_fe_m.Solve();

      // Determine the current size of the linear system
      int prob_size = Mfem_fe_m.GetProblemSize();

      // Write fields to disk for VisIt
      if ( visit )
      {
         Mfem_fe_m.WriteVisItFields(it);
      }

      // Send the solution by socket to a GLVis server.
      if (visualization)
      {
         Mfem_fe_m.DisplayToGLVis();
      }

      if (Mpi::Root())
      {
         cout << "AMR iteration " << it << " complete." << endl;
      }

      // Check stopping criteria
      if (prob_size > max_dofs)
      {
         if (Mpi::Root())
         {
            cout << "Reached maximum number of dofs, exiting..." << endl;
         }
         break;
      }
      if ( it == maxit )
      {
         break;
      }

      // Wait for user input. Ask every 10th iteration.
      char c = 'c';
      if (Mpi::Root() && (it % 10 == 0))
      {
         cout << "press (q)uit or (c)ontinue --> " << flush;
         cin >> c;
      }
      MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

      if (c != 'c')
      {
         break;
      }

      // Estimate element errors using the Zienkiewicz-Zhu error estimator.
      Vector errors(pmesh.GetNE());
      Mfem_fe_m.GetErrorEstimates(errors);

      real_t local_max_err = errors.Max();
      real_t global_max_err;
      MPI_Allreduce(&local_max_err, &global_max_err, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh.GetComm());

      // Refine the elements whose error is larger than a fraction of the
      // maximum element error.
      const real_t frac = 0.5;
      real_t threshold = frac * global_max_err;
      if (Mpi::Root()) { cout << "Refining ..." << endl; }
      pmesh.RefineByError(errors, threshold);

      // Update the magnetostatic solver to reflect the new state of the mesh.
      Mfem_fe_m.Update();

      if (pmesh.Nonconforming() && Mpi::WorldSize() > 1)
      {
         if (Mpi::Root()) { cout << "Rebalancing ..." << endl; }
         pmesh.Rebalance();

         // Update again after rebalancing
         Mfem_fe_m.Update();
      }
   }

   delete muInvCoef;

   return 0;
}

// Print the Volta ascii logo to the given ostream
void display_banner(ostream & os)
{
   os << "  ______________________________   " << endl
      << "      Fusion Energy Magnetics   " << endl << flush;
}

// The Permeability is a required coefficient which may be defined in
// various ways so we'll determine the appropriate coefficient type here.
Coefficient *
SetupInvPermeabilityCoefficient()
{
   Coefficient * coef = NULL;

   if ( pw_mu_.Size() > 0 )
   {
      pw_mu_inv_.SetSize(pw_mu_.Size());
      for (int i = 0; i < pw_mu_.Size(); i++)
      {
         MFEM_ASSERT( pw_mu_[i] > 0.0, "permeability values must be positive" );
         pw_mu_inv_[i] = 1.0/pw_mu_[i];
      }
      coef = new PWConstCoefficient(pw_mu_inv_);
   }
   else
   {
      coef = new ConstantCoefficient(1.0/mu0_);
      //cout<<" permeability coefficient "<<1.0/mu0_<<endl;
   }

   return coef;
}




void ComputeCurrentDensityOnSubMesh(int order,
                                    bool visualization,
				    Coefficient &sigmaCoef, 
                                    const Array<int> &phi0_attr,
                                    const Array<int> &phi1_attr,
                                    const Array<int> &jn_zero_attr,
                                    ParGridFunction &phi_h1,
                                    ParGridFunction &j_cond,
                                    ParGridFunction &j_l2)
{
   // Extract the finite element space and mesh on which j_cond is defined
   ParFiniteElementSpace &fes_cond_rt = *j_cond.ParFESpace();
   ParFiniteElementSpace &fes_cond_l2 = *j_l2.ParFESpace();
   ParMesh &pmesh_cond = *fes_cond_rt.GetParMesh();
   // the following is extracting potential, not current
   ParFiniteElementSpace &fes_cond_h1 = *phi_h1.ParFESpace();
   //ParMesh &pmesh_cond = *fes_cond.GetParMesh();
   
   int myid = fes_cond_rt.GetMyRank();
   //int myid = fes_cond.GetMyRank();
   int dim  = pmesh_cond.Dimension();

   // Define a parallel finite element space on the SubMesh. Here we use the
   // H1 finite elements for the electrostatic potential.

   // Define the conductivity coefficient and the boundaries associated with the
   // fixed potentials phi0 and phi1 which will drive the current.
   
   int nelem = pmesh_cond.GetNE();
   cout<<"number of coil elements "<<nelem<<endl;

   Array<int> ess_bdr_phi(pmesh_cond.bdr_attributes.Max());
   Array<int> ess_bdr_j(pmesh_cond.bdr_attributes.Max());
   Array<int> ess_bdr_tdof_phi;
   ess_bdr_phi = 0;
   ess_bdr_j   = 0;
   for (int i=0; i<phi0_attr.Size(); i++)
   {
      ess_bdr_phi[phi0_attr[i]-1] = 1;
   }
   for (int i=0; i<phi1_attr.Size(); i++)
   {
      ess_bdr_phi[phi1_attr[i]-1] = 1;
   }
   for (int i=0; i<jn_zero_attr.Size(); i++)
   {
      ess_bdr_j[jn_zero_attr[i]-1] = 1;
   }
   fes_cond_h1.GetEssentialTrueDofs(ess_bdr_phi, ess_bdr_tdof_phi);

   // Setup the bilinear form corresponding to -Div(sigma Grad phi)
   ParBilinearForm a_h1(&fes_cond_h1);
   a_h1.AddDomainIntegrator(new DiffusionIntegrator(sigmaCoef));
   a_h1.Assemble();

   // Set the r.h.s. to zero
   ParLinearForm b_h1(&fes_cond_h1);
   b_h1 = 0.0;

   // Setup the boundary conditions on phi
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   //ParGridFunction phi_h1(&fes_cond_h1);
   phi_h1 = 0.0;

   Array<int> bdr0(pmesh_cond.bdr_attributes.Max()); bdr0 = 0;
   for (int i=0; i<phi0_attr.Size(); i++)
   {
      bdr0[phi0_attr[i]-1] = 1;
   }
   phi_h1.ProjectBdrCoefficient(zero, bdr0);

   Array<int> bdr1(pmesh_cond.bdr_attributes.Max()); bdr1 = 0;
   for (int i=0; i<phi1_attr.Size(); i++)
   {
      bdr1[phi1_attr[i]-1] = 1;
   }
   phi_h1.ProjectBdrCoefficient(one, bdr1);

   // Solve the linear system using algebraic multigrid
   {
      if (myid == 0)
      {
         cout << "\nSolving for electric potential "
              << "using CG with AMG" << endl;
      }
      OperatorPtr A;
      Vector B, X;
      a_h1.FormLinearSystem(ess_bdr_tdof_phi, phi_h1, b_h1, A, X, B);

      HypreBoomerAMG prec;
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(1);
      cg.SetPreconditioner(prec);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      a_h1.RecoverFEMSolution(X, b_h1, phi_h1);
   }
   if (myid == 0)
   {
      ofstream sol_ofs("phi.gf");
      sol_ofs.precision(8);
      phi_h1.Save(sol_ofs);
   }
   if (visualization)
   {
      int num_procs = fes_cond_h1.GetNRanks();
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream port_sock(vishost, visport);
      port_sock << "parallel " << num_procs << " " << myid << "\n";
      port_sock.precision(8);
      port_sock << "solution\n" << pmesh_cond << phi_h1
                << "window_title 'Conductor Potential'"
                << "window_geometry 0 0 400 350" << flush;
   }

   // Solve for the current density J = -sigma Grad phi with boundary conditions phi=0 and phi=1.
   // We compute J in two difference spaces, H(div) and L2. 
   // Frist j_cond which is in H(div) so we need an RT mass matrix
   ParBilinearForm m_rt(&fes_cond_rt);
   m_rt.AddDomainIntegrator(new VectorFEMassIntegrator);
   m_rt.Assemble();

   // Next construct the L2 mass matrix for computing j_l2
   ParBilinearForm m_l2(&fes_cond_l2);
   m_l2.AddDomainIntegrator(new VectorMassIntegrator);
   m_l2.Assemble();

   // Assemble the (sigma Grad phi) operator
   ParMixedBilinearForm d_h1(&fes_cond_h1, &fes_cond_rt);
   d_h1.AddDomainIntegrator(new MixedVectorGradientIntegrator(sigmaCoef));
   d_h1.Assemble();

   // Compute the r.h.s, b_rt = sigma E = -sigma Grad phi
   ParLinearForm b_rt(&fes_cond_rt);
   d_h1.Mult(phi_h1, b_rt);
   b_rt *= -1.0;

   // Assemble the (sigma Grad phi) operator in l2
   ParMixedBilinearForm d_h1l2(&fes_cond_h1, &fes_cond_l2);
   //ParDiscreteLinearOperator d_h1l2(&fes_cond_h1, &fes_cond_l2);
   d_h1l2.AddDomainIntegrator(new GradientIntegrator(sigmaCoef));
   d_h1l2.Assemble();
   // Compute the r.h.s, b_rt = sigma E = -sigma Grad phi in L2
   ParLinearForm b_l2(&fes_cond_l2);
   d_h1l2.Mult(phi_h1, b_l2);
   b_l2 *= -1.0;
   //b_l2 *= 1.0*2.0*M_PI*10;
   
   // Apply the necessary boundary conditions and solve for J in H(div)
   HYPRE_BigInt glb_size_rt = fes_cond_rt.GlobalTrueVSize();
   HYPRE_BigInt glb_size_l2 = fes_cond_l2.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "\nSolving for current density in H(Div) "
           << "using diagonally scaled CG" << endl;
      cout << "Size of linear system: "
           << glb_size_rt << endl;
      cout << "Size of linear system in L2: "
           << glb_size_l2 << endl;
   }
   Array<int> ess_bdr_tdof_rt;
   Array<int> ess_bdr_tdof_l2;
   OperatorPtr M;
   Vector B, X;

   fes_cond_rt.GetEssentialTrueDofs(ess_bdr_j, ess_bdr_tdof_rt);
   fes_cond_l2.GetEssentialTrueDofs(ess_bdr_j, ess_bdr_tdof_l2);

   j_cond = 0.0;
   m_rt.FormLinearSystem(ess_bdr_tdof_rt, j_cond, b_rt, M, X, B);
   
   
   OperatorPtr M2;
   Vector B2, X2;
   j_l2 = 0.0;
   m_l2.FormLinearSystem(ess_bdr_tdof_l2, j_l2, b_l2, M2, X2, B2);

   HypreDiagScale prec;

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(prec);
   cg.SetOperator(*M);
   cg.Mult(B, X);
   m_rt.RecoverFEMSolution(X, b_rt, j_cond);

   cg.SetOperator(*M2);
   cg.Mult(B2, X2);
   m_l2.RecoverFEMSolution(X2, b_l2, j_l2);
}
