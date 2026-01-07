//
//
#include "heat.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

//using namespace std;
//using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();


   //  Parse command-line options.
   const char *mesh_file = "heatcoil.g";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool algebraic_ceed = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
#ifdef MFEM_USE_CEED
   args.AddOption(&algebraic_ceed, "-a", "--algebraic", "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   //  Enable hardware devices such as GPUs, and programming models such as  CUDA if needed
   Device device(device_config);
   device.Print();

   // Read the mesh from the given mesh file. 
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   // define l2 space
   L2_FECollection l2c(0, mesh.Dimension());
   ParFiniteElementSpace l2_fes(&pmesh, &l2c);
   GridFunction q_elem(&l2_fes);
   q_elem = 0.;


   // Define a finite element space on the mesh. 
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (pmesh.GetNodes())
   {
      fec = pmesh.GetNodes()->OwnFEC();
      delete_fec = false;
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   ParFiniteElementSpace fespace(&pmesh, fec);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }



   const double k_coil   = 200.0;    // thermal conductivity
   const double k_case   = 20.0;
   //98 is coil 112 is case


   const double h_val   = 8.33e-4/1.0;    // convection coefficient
   const double T_inf   = 300.0;   // ambient
   //const double q_val = 1.0;     // volumetric source q (set to 0.0 if none)

   // Coefficients
   //ConstantCoefficient kcoef(k_val);

   Vector k_vals(pmesh.attributes.Max());  // size = max attribute id
   k_vals = 0.0;                          // default (unused attrs) = 0

   // attributes are 1-based, vector is 0-based:
   // attribute i  ->  k_vals[i-1]
   k_vals[98 - 1] = k_coil;   // coil region
   k_vals[112 - 1] = k_case;  // case region
   PWConstCoefficient kcoef(k_vals);

   ConstantCoefficient hcoef(h_val);
   ConstantCoefficient hTinf(h_val * T_inf);

   //const string qfile = "q.txt";              // file path
   //double q_val = ReadFirstValue(qfile, 0.0);      // default to 0 if not found
   //cout<<"source value "<<q_val<<endl;
   //q_val = 1.0;

   bool ids_one_based = true;
    //Reading from first file: first batch of elements 
    ElemValueMap batch1 = ReadElemValues("data/Coil_Heating_cells_threshold_1e2_columns_wsink.txt");
    FillL2FromMap(pmesh, batch1, ids_one_based, q_elem);

    //Reading from second file: remaining elements 
    ElemValueMap batch2 = ReadElemValues("data/Case_Heating_cells_threshold_1e3_columns.txt");
    FillL2FromMap(pmesh, batch2, ids_one_based, q_elem);
    GridFunctionCoefficient qcoef(&q_elem);



   //ConstantCoefficient Tbc(T_dir);

   //  Determine the list of true essential boundary dofs.
   //  In our case, it's all Robin BC.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 0;
      //ess_bdr[4] = ess_bdr[5] = ess_bdr[6] = ess_bdr[7] = 1;
      // Apply boundary conditions on all external boundaries:
      //mesh.MarkExternalBoundaries(ess_bdr);
      // Boundary conditions can also be applied based on named attributes:
      // mesh.MarkNamedBoundaries(set_name, ess_bdr)

      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   Array<int> robin_bdr(pmesh.bdr_attributes.Max()); robin_bdr = 0;
   //robin_bdr[0] = robin_bdr[1] = robin_bdr[2] = robin_bdr[3] = 1;
   robin_bdr = 1;


   //  Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system , which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(qcoef));
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(hTinf), robin_bdr); // rhs from Robin

   b.Assemble();

   //  Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

    // Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   ParBilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   if (fa)
   {
      a.SetAssemblyLevel(AssemblyLevel::FULL);
      // Sort the matrix column indices when running on GPU or with OpenMP (i.e.
      // when Device::IsEnabled() returns true). This makes the results
      // bit-for-bit deterministic at the cost of somewhat longer run time.
      a.EnableSparseMatrixSorting(Device::IsEnabled());
   }
   a.AddDomainIntegrator(new DiffusionIntegrator(kcoef));
   a.AddBoundaryIntegrator(new BoundaryMassIntegrator(hcoef), robin_bdr);


   //  Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   // 11. Solve the linear system A X = B.
   Solver *prec = NULL;
   if (pa)
   {
      if (UsesTensorBasis(fespace))
      {
         if (algebraic_ceed)
         {
            prec = new ceed::AlgebraicSolver(a, ess_tdof_list);
         }
         else
         {
            prec = new OperatorJacobiSmoother(a, ess_tdof_list);
         }
      }
   }
   else
   {
      prec = new HypreBoomerAMG;
   }
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   if (prec) { cg.SetPreconditioner(*prec); }
   cg.SetOperator(*A);
   cg.Mult(B, X);
   delete prec;


   /* 
   if (!pa)
   {
#ifndef MFEM_USE_SUITESPARSE
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M((SparseMatrix&)(*A));
      //GSSmoother M((SparseMatrix&)(*A), 0, 2);
      PCG(*A, M, B, X, 1, 600, 1e-12, 0.0);
#else
      // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(*A);
      umf_solver.Mult(B, X);
#endif
   }
   else
   {
      if (UsesTensorBasis(fespace))
      {
         if (algebraic_ceed)
         {
            ceed::AlgebraicSolver M(a, ess_tdof_list);
            PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
         }
         else
         {
            OperatorJacobiSmoother M(a, ess_tdof_list);
            PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
         }
      }
      else
      {
         CG(*A, B, X, 1, 400, 1e-12, 0.0);
      }
   }
   */
   // 12. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);


   //checking residual
   Vector r(B.Size()), Ax(B.Size());
   A->Mult(X, Ax);
   subtract(B, Ax, r);
   cout << "rel_res = " << r.Norml2() / B.Norml2() << endl;

   double rel_res = r.Norml2() / std::max(B.Norml2(), 1e-30);
   cout << "TRUE relative residual ||B-AX||/||B|| = "
     << rel_res << endl;
   // 13. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
   }


   VisItDataCollection visit_dc("ex1", &mesh);
   visit_dc.RegisterField("temperature", &x);
   visit_dc.SetPrecision(8);               // optional: nicer ASCII precision
   visit_dc.SetPrefixPath("out");              // optional
   // save older version of mesh
   AttributeSets &as     = mesh.attribute_sets;
   AttributeSets &bdr_as = mesh.bdr_attribute_sets;

   if (as.SetsExist())
   {
     for (const auto &name : as.GetAttributeSetNames()) { as.DeleteAttributeSet(name); }
   }
   if (bdr_as.SetsExist())
   {
     for (const auto &name : bdr_as.GetAttributeSetNames()) { bdr_as.DeleteAttributeSet(name); }
   }


   visit_dc.Save();
   // 15. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}




