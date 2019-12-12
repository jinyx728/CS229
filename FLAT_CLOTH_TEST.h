//#####################################################################
// Copyright 2019, Jane Wu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class FLAT_CLOTH_TEST
//#####################################################################
#ifndef __FLAT_CLOTH_TEST__
#define __FLAT_CLOTH_TEST__

#include <PhysBAM_Geometry/Topology_Based_Geometry/TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Surface_Mesh_Level_Sets/FAST_MARCHING_METHOD_SURFACE_MESH.h>
#include <PhysBAM_Dynamics/Morphing/LAPLACE_SURFACE_MORPH.h>
#include <fstream>

namespace PhysBAM{
template<typename T>
class FLAT_CLOTH_TEST
{
    typedef VECTOR<T,3> TV;
public:

    TRIANGULATED_SURFACE<T>* triangulated_surface;
    ARRAY<VECTOR<T,2> > vt;
    ARRAY<int> seed_indices;
    ARRAY<T> phi;
    ARRAY<T> in_dx;
    ARRAY<T> in_dy;

    std::string output_phi_filename;
    std::string output_vt_x_filename;
    std::string output_vt_y_filename;

    virtual ~FLAT_CLOTH_TEST()
    {
        delete triangulated_surface;
    }

FLAT_CLOTH_TEST(std::string input_vt_filename, std::string input_disp_filename, std::string out_folder)
{
    std::string input_mesh_filename="/data/jwu/PhysBAM/Private_Projects/cloth_texture/tests/cloth_front/flat_tshirt_front.tri.gz";

    output_phi_filename = out_folder+"/out_phi.txt";
    output_vt_x_filename = out_folder+"/out_vt_x.txt";
    output_vt_y_filename = out_folder+"/out_vt_y.txt";

    FILE_UTILITIES::Create_From_File<T>(input_mesh_filename,triangulated_surface);
    triangulated_surface->Initialize_Hierarchy();
    triangulated_surface->Update_Triangle_List();
    triangulated_surface->Update_Vertex_Normals();
    triangulated_surface->mesh.Initialize_Neighbor_Nodes();
    triangulated_surface->mesh.Initialize_Incident_Elements();

    //Read in ground truth texture coordinates
    std::ifstream vt_file(input_vt_filename);
    std::string label;
    T a, b;
    while (vt_file >> label >> a >> b){
        vt.Append(VECTOR<T,2>(a,b));
    }
   
    //Read in texture coordinate displacements after ray-intersection method
    std::ifstream disp_file(input_disp_filename);
    size_t count = 0;
    size_t idx = 1;
    ARRAY<T> in_dx_full;
    ARRAY<T> in_dy_full;
    ARRAY<T> phi_full; //Hard coded for now...
    T prev_phi = 1;
    while (disp_file >> a >> b){
        in_dx_full.Append(a);
        in_dy_full.Append(b);
        //Set starting phi values
        if (a==0.0 && b ==0.0) {
            count += 1;
            phi_full.Append(1);
        }
        else {
            phi_full.Append(-1);
        }
        prev_phi = phi_full(idx);
        idx += 1;
    }
    //Read in front vertices
    std::string front_vertices_filename="/data/jwu/PhysBAM/Private_Projects/cloth_on_virtual_body/Learning/shared_data/front_vertices.txt";
    std::ifstream front_vertices_file(front_vertices_filename);
    T front_id;
    while (front_vertices_file >> front_id) {
        const int i = (int)front_id + 1; //Zero indexing
        phi.Append(phi_full(i));
        in_dx.Append(in_dx_full(i));
        in_dy.Append(in_dy_full(i));
    }
    
    //Set final seed indices
    SEGMENT_MESH& segment_mesh = triangulated_surface->Get_Segment_Mesh();
    ARRAY<T> is_seed(triangulated_surface->particles.array_collection->Size());
    for (size_t i = 1; i <= segment_mesh.elements.m; i++) {
        const int start = segment_mesh.elements(i)(1);
        const int end = segment_mesh.elements(i)(2);
        const TV start_pt = triangulated_surface->particles.X(start);
        const TV end_pt = triangulated_surface->particles.X(end);
        if (sgn(phi(start)) != sgn(phi(end))) {
            //Use 1/2 edge length as phi values
            const T dist = sqrt(pow(end_pt(1)-start_pt(1),2) + pow(end_pt(2)-start_pt(2),2));
            phi(start) = phi(start)*dist/2;
            phi(end) = phi(end)*dist/2;
            is_seed(start) = 1;
            is_seed(end) = 1;
        }
    }

    for (size_t i = 1; i <= triangulated_surface->particles.array_collection->Size(); i++) {
        if (is_seed(i) == 1)
            seed_indices.Append(i);
    }
}

//#####################################################################
// Function Extrapolate
//#####################################################################
void Extrapolate()
{
    T stopping_distance = 0;
    FAST_MARCHING_METHOD_SURFACE_MESH<TV> fmm_x(*triangulated_surface, *triangulated_surface->vertex_normals);
    fmm_x.Fast_Marching_Method(phi,stopping_distance,&seed_indices);

    //Extrapolate x
    LAPLACE_SURFACE_MORPH<T> laplace_surface_morph_x(*triangulated_surface, in_dx);
    //Set boundary conditions
    for (size_t i = 1; i <= triangulated_surface->particles.array_collection->Size(); i++) {
        if (phi(i) <= 0)
            laplace_surface_morph_x.psi_D_nodes(i) = true;
    }
    laplace_surface_morph_x.Solve();

    //Extrapolate y
    LAPLACE_SURFACE_MORPH<T> laplace_surface_morph_y(*triangulated_surface, in_dy);
    //Set boundary conditions
    for (size_t i = 1; i <= triangulated_surface->particles.array_collection->Size(); i++) {
        if (phi(i) <= 0)
            laplace_surface_morph_y.psi_D_nodes(i) = true;
    }
    laplace_surface_morph_y.Solve();
}

//#####################################################################
// Function Write_Outputs
//#####################################################################
void Write_Outputs()
{
    // Write output phi
    std::ostream* output=FILE_UTILITIES::Safe_Open_Output(output_phi_filename,false,false);
    for (size_t i = 1; i <= phi.Size(); i++)
        *output << phi(i) << "\n";
    delete output;

    // Write output tcs
    output=FILE_UTILITIES::Safe_Open_Output(output_vt_x_filename,false,false);
    for (size_t i = 1; i <= in_dx.Size(); i++)
        *output << in_dx(i) << "\n";
    delete output;

   // Write output tcs
    output=FILE_UTILITIES::Safe_Open_Output(output_vt_y_filename,false,false);
    for (size_t i = 1; i <= in_dy.Size(); i++)
        *output << in_dy(i) << "\n";
    delete output;
}

private:

int sgn(double v) {
      return (v < 0) ? -1 : ((v > 0) ? 1 : 0);
}
};
}
#endif
