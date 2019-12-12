//#####################################################################
// Copyright 2019, Jane Wu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class GRID_TEST
//#####################################################################
#ifndef __GRID_TEST__
#define __GRID_TEST__

#include <PhysBAM_Geometry/Topology_Based_Geometry/TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Surface_Mesh_Level_Sets/FAST_MARCHING_METHOD_SURFACE_MESH.h>
#include <PhysBAM_Dynamics/Morphing/LAPLACE_SURFACE_MORPH.h>

namespace PhysBAM{
template<typename T>
class GRID_TEST
{
    typedef VECTOR<T,3> TV;
public:

    TRIANGULATED_SURFACE<T>* triangulated_surface;
    ARRAY<int> seed_indices;
    ARRAY<T> phi_final;

    std::string output_phi_filename;
    std::string output_vt_filename;

    virtual ~GRID_TEST()
    {
        delete triangulated_surface;
    }

GRID_TEST(std::string input_mesh_filename, std::string out_folder, bool use_circle)
{
    output_phi_filename = out_folder+"/out_phi.txt";
    output_vt_filename = out_folder+"/out_vt.txt";

    FILE_UTILITIES::Create_From_File<T>(input_mesh_filename,triangulated_surface);
    triangulated_surface->Initialize_Hierarchy();
    triangulated_surface->Update_Triangle_List();
    triangulated_surface->Update_Vertex_Normals();
    triangulated_surface->mesh.Initialize_Neighbor_Nodes();
    triangulated_surface->mesh.Initialize_Incident_Elements();

    int dim = 100;
    ARRAY<T> phi(triangulated_surface->particles.array_collection->Size());

    // L/R split
    if (use_circle == false) {
    for (int i = 1; i <= triangulated_surface->particles.array_collection->Size(); i++) {
        if ((i-1) % dim >= 49 && (i-1) % dim <= 51)
            seed_indices.Append(i);
        if ((i-1) % dim < 50) {
            phi(i) = -1;
        }
        else if ((i-1) % dim == 50)
            phi(i) = 0;
        else
            phi(i) = 1;
    }}
    // Circle
    else {
    T rad = 30;
    for (size_t i = 1; i <= triangulated_surface->particles.array_collection->Size(); i++) {
        const TV pt = triangulated_surface->particles.X(i);
        const T x = pt(1);
        const T y = pt(2);
        if (pow(x-dim/2,2)+pow(y-dim/2,2) <= pow(rad,2)) {
            phi(i) = -10;
        }
        else
            phi(i) = 10;
    }
    SEGMENT_MESH& segment_mesh = triangulated_surface->Get_Segment_Mesh();
    ARRAY<T> is_seed(triangulated_surface->particles.array_collection->Size());
    for (size_t i = 1; i <= segment_mesh.elements.m; i++) {
        const int start = segment_mesh.elements(i)(1);
        const int end = segment_mesh.elements(i)(2);
        const TV start_pt = triangulated_surface->particles.X(start);
        const TV end_pt = triangulated_surface->particles.X(end);
        if (phi(start) != phi(end)) {
            phi(start) = sqrt(pow(start_pt(1)-dim/2,2) + pow(start_pt(2)-dim/2,2)) - rad;
            phi(end) = sqrt(pow(end_pt(1)-dim/2,2) + pow(end_pt(2)-dim/2,2)) - rad;
            is_seed(start) = 1;
            is_seed(end) = 1;
        }
    }
    for (size_t i = 1; i <= triangulated_surface->particles.array_collection->Size(); i++) {
        if (is_seed(i) == 1)
            seed_indices.Append(i);
    }}
    phi_final = phi;
}

//#####################################################################
// Function Extrapolate
//#####################################################################
void Extrapolate(ARRAY<T>& u_input)
{
    T stopping_distance=0;
 
    FAST_MARCHING_METHOD_SURFACE_MESH<TV> fmm_x(*triangulated_surface,(*triangulated_surface->vertex_normals));
    fmm_x.Fast_Marching_Method(phi_final,stopping_distance,&seed_indices);

    LAPLACE_SURFACE_MORPH<T> laplace_surface_morph(*triangulated_surface, u_input);
    //Set boundary conditions
    for (size_t i = 1; i <= triangulated_surface->particles.array_collection->Size(); i++) {
        if (phi_final(i) <= 0)
            laplace_surface_morph.psi_D_nodes(i) = true;
    }
    laplace_surface_morph.Solve();
}

//#####################################################################
// Function Write_Outputs
//#####################################################################
void Write_Outputs(ARRAY<T>& u_input)
{
    // Write output phi
    std::ostream* output=FILE_UTILITIES::Safe_Open_Output(output_phi_filename,false,false);
    for (size_t i = 1; i <= phi_final.Size(); i++) {
        *output << phi_final(i) << "\n";
    }
    delete output;

    // Write output tcs
    output=FILE_UTILITIES::Safe_Open_Output(output_vt_filename,false,false);
    for (size_t i = 1; i <= u_input.Size(); i++)
        *output << u_input(i) << "\n";
    delete output;
}
};
}
#endif
