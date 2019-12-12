//#####################################################################
// Copyright 2019, Jane Wu.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include <PhysBAM_Tools/Log/LOG.h>
#include <PhysBAM_Tools/Parsing/PARSE_ARGS.h>
#include <PhysBAM_Tools/Read_Write/Grids_Uniform_Arrays/READ_WRITE_ARRAYS.h>
#include <PhysBAM_Tools/Read_Write/Arrays/READ_WRITE_ARRAY.h>
#include <PhysBAM_Tools/Read_Write/Utilities/FILE_UTILITIES.h>
#include <PhysBAM_Tools/Utilities/PROCESS_UTILITIES.h>
#include <PhysBAM_Geometry/Basic_Geometry/TETRAHEDRON.h>
#include <PhysBAM_Geometry/Read_Write/Geometry/READ_WRITE_TRIANGULATED_SURFACE.h>
#include <PhysBAM_Geometry/Surface_Mesh_Level_Sets/FAST_MARCHING_METHOD_SURFACE_MESH.h>
#include <PhysBAM_Geometry/Surface_Mesh_Level_Sets/EXTRAPOLATION_SURFACE_MESH.h>
#include <PhysBAM_Geometry/Topology_Based_Geometry/TRIANGULATED_SURFACE.h>
#include <PhysBAM_Dynamics/Geometry/GENERAL_GEOMETRY_FORWARD.h>
#include <PhysBAM_Dynamics/Morphing/LAPLACE_SURFACE_MORPH.h>
#include <PhysBAM_Dynamics/Particles/PARTICLES_FORWARD.h>
#include <PhysBAM_Solids/PhysBAM_Deformables/Particles/PARTICLES.h>
#include "GRID_TEST.h"
#include "FLAT_CLOTH_TEST.h"
#include <fstream>
#include <iostream>

using namespace PhysBAM;

int sgn(double v) {
      return (v < 0) ? -1 : ((v > 0) ? 1 : 0);
}

template<class T, class RW> void Fast_Marching(TRIANGULATED_SURFACE<T>* triangulated_surface, ARRAY<T>& phi, ARRAY<int>& seed_indices,
                                               ARRAY<T>& in_dx, ARRAY<T>& in_dy, ARRAY<T>& in_dz, std::string out_folder, std::string frame)
{
    typedef VECTOR<T,3> TV;

    // Signed distance field
    T stopping_distance = 0;
    FAST_MARCHING_METHOD_SURFACE_MESH<TV> fmm_x(*triangulated_surface, *triangulated_surface->vertex_normals);
    fmm_x.Fast_Marching_Method(phi,stopping_distance,&seed_indices);

    //Get front triangles
    std::string front_vertices_filename="/phoenix/yxjin/shared_data_highres/front_vertices.txt";
    std::ifstream front_vertices_file(front_vertices_filename);
    T front_id;
    ARRAY<int> front_indices(triangulated_surface->particles.array_collection->Size());
    int count = 0;
    while (front_vertices_file >> front_id) {
        const int i = (int)front_id + 1; //Zero indexing
        front_indices(i) = 1;
        count += 1;
    }
    std::cout << "Num front indices: " << count << std::endl;

    //Extrapolate x
    LAPLACE_SURFACE_MORPH<T> laplace_surface_morph_x(*triangulated_surface, in_dx);
    //Set boundary conditions
    for (int i = 1; i <= triangulated_surface->particles.array_collection->Size(); i++) {
        if (phi(i) <= 0)
            laplace_surface_morph_x.psi_D_nodes(i) = true;
    }
    laplace_surface_morph_x.Solve();

    //Extrapolate y
    LAPLACE_SURFACE_MORPH<T> laplace_surface_morph_y(*triangulated_surface, in_dy);
    //Set boundary conditions
    for (int i = 1; i <= triangulated_surface->particles.array_collection->Size(); i++) {
        if (phi(i) <= 0)
            laplace_surface_morph_y.psi_D_nodes(i) = true;
    }
    laplace_surface_morph_y.Solve();

    //Extrapolate z
    LAPLACE_SURFACE_MORPH<T> laplace_surface_morph_z(*triangulated_surface, in_dz);
    //Set boundary conditions
    for (int i = 1; i <= triangulated_surface->particles.array_collection->Size(); i++) {
        if (phi(i) <= 0)
            laplace_surface_morph_z.psi_D_nodes(i) = true;
    }
    laplace_surface_morph_z.Solve();
    printf("Done with laplace solve\n");

    /*
    //Reset back vertices
    for (int i = 1; i <= back_indices.Size(); i++) {
        in_dx(back_indices(i)) = 0;
        in_dy(back_indices(i)) = 0;
    }*/

    std::string output_phi_filename = out_folder + "/reconstruct_filled_phi.txt";
    std::string output_obj_filename = out_folder + "/reconstruct_filled.obj";
  
    // Write output phi
    std::ostream* output=FILE_UTILITIES::Safe_Open_Output(output_phi_filename,false,false);
    for (int i = 1; i <= phi.Size(); i++)
        *output << phi(i) << "\n";
    delete output;

    // Write out obj
    output=FILE_UTILITIES::Safe_Open_Output(output_obj_filename,false);
    std::string header("# simple obj file format:\n"
        "#   # vertex at coordinates (x,y,z)\n"
        "#   v x y z\n"
        "#   # triangle with vertices a,b,c\n"
        "#   f a b c\n"
        "#   # vertices are indexed starting from 1\n"
        "\n");
    (*output)<<header;

    for(int p=1;p<=triangulated_surface->particles.array_collection->Size();p++)
        (*output)<<STRING_UTILITIES::string_sprintf("v %lg %lg %lg\n",
                                                    triangulated_surface->particles.X(p)[1] + in_dx(p),
                                                    triangulated_surface->particles.X(p)[2] + in_dy(p),
                                                    triangulated_surface->particles.X(p)[3] + in_dz(p));

    for(int e=1;e<=triangulated_surface->mesh.elements.m;e++) {
        if (front_indices(triangulated_surface->mesh.elements(e)[1]) == 1 &&
            front_indices(triangulated_surface->mesh.elements(e)[2]) == 1 &&
            front_indices(triangulated_surface->mesh.elements(e)[3]) == 1)
            (*output)<<STRING_UTILITIES::string_sprintf("f %d %d %d\n",
                                                        triangulated_surface->mesh.elements(e)[1],
                                                        triangulated_surface->mesh.elements(e)[2],
                                                        triangulated_surface->mesh.elements(e)[3]);
    }
    delete output;
}

template<class T,class RW> void Flood_Fill(std::string input_mesh_filename, std::string input_pred_filename, std::string input_idx_filename, std::string out_folder, std::string frame)
{
    typedef VECTOR<T,3> TV; 

    //Read in gt cloth mesh
    std::cout << "reading " << input_mesh_filename << std::endl;
    TRIANGULATED_SURFACE<T>* triangulated_surface=0;
    FILE_UTILITIES::Create_From_File<RW>(input_mesh_filename,triangulated_surface);
    triangulated_surface->Update_Vertex_Normals();
    triangulated_surface->mesh.Initialize_Neighbor_Nodes();
    triangulated_surface->mesh.Initialize_Incident_Elements();

    //Read in reconstructed cloth mesh
    std::cout << "reading " << input_pred_filename << std::endl;
    TRIANGULATED_SURFACE<T>* reconstructed_surface=0;
    FILE_UTILITIES::Create_From_File<RW>(input_pred_filename,reconstructed_surface);

    //Sanity check
    std::cout << "PD vertices: " << triangulated_surface->particles.array_collection->Size() << std::endl;
    std::cout << "RE vertices: " << reconstructed_surface->particles.array_collection->Size() << std::endl;
    
    //Read in texture coordinate displacements after ray-intersection method
    std::ifstream idx_file(input_idx_filename);
    T a;
    int b;
    int idx = 1;
    ARRAY<T> in_dx(triangulated_surface->particles.array_collection->Size());
    ARRAY<T> in_dy(triangulated_surface->particles.array_collection->Size());
    ARRAY<T> in_dz(triangulated_surface->particles.array_collection->Size()); 
    ARRAY<T> phi(triangulated_surface->particles.array_collection->Size());
    for (int i = 1; i <= triangulated_surface->particles.array_collection->Size(); i++)
        phi(i) = 1;
    while (idx_file >> a){
        b = (int) a;
        in_dx(b+1) = reconstructed_surface->particles.X(b+1)[1] - triangulated_surface->particles.X(b+1)[1];
        in_dy(b+1) = reconstructed_surface->particles.X(b+1)[2] - triangulated_surface->particles.X(b+1)[2];
        in_dz(b+1) = reconstructed_surface->particles.X(b+1)[3] - triangulated_surface->particles.X(b+1)[3];
        phi(b+1) = -1;
        idx += 1;
    }

    std::cout << "Init num reconstructed vertices: " << idx -1 << std::endl;

    //Set final seed indices
    ARRAY<int> seed_indices;
    SEGMENT_MESH& segment_mesh = triangulated_surface->Get_Segment_Mesh();
    ARRAY<T> is_seed(triangulated_surface->particles.array_collection->Size());
    std::cout << "Num segments: " << segment_mesh.elements.m << std::endl;
    for (int i = 1; i <= segment_mesh.elements.m; i++) {
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
    for (int i = 1; i <= triangulated_surface->particles.array_collection->Size(); i++) {
        if (is_seed(i) == 1)
            seed_indices.Append(i);
    }
    std::cout << "Seed idx: " << seed_indices.Size() << std::endl;

    Fast_Marching<T,RW>(triangulated_surface, phi, seed_indices, in_dx, in_dy, in_dz, out_folder, frame);
}

int main(int argc,char *argv[])
{
    LOG::Initialize_Logging(false,false,1<<30,true,1);
    printf("\n");

    PROCESS_UTILITIES::Set_Floating_Point_Exception_Handling(true);
    Initialize_Read_Write_General_Structures();

    PARSE_ARGS parse_args;
    parse_args.Set_Extra_Arguments(1,"<pd_mesh>","<pd_mesh> inferred cloth mesh");
    parse_args.Set_Extra_Arguments(2,"<re_mesh>","<re_mesh> reconstructed mesh to be extrapolated from");
    parse_args.Set_Extra_Arguments(3,"<re_idx>","<re_idx> indices of reconstructed vertices");
    parse_args.Set_Extra_Arguments(4,"<output folder>","<output folder> output folder");
    parse_args.Set_Extra_Arguments(5,"<frame>","<frame> pose frame number (for output filename)");
    parse_args.Parse(argc, argv);

    std::string input_mesh_filename=parse_args.Extra_Arg(1);
    std::string input_pred_filename=parse_args.Extra_Arg(2);
    std::string input_idx_filename=parse_args.Extra_Arg(3);
    std::string out_folder=parse_args.Extra_Arg(4);
    std::string frame = parse_args.Extra_Arg(5);
 
    Flood_Fill<float,float>(input_mesh_filename, input_pred_filename, input_idx_filename, out_folder, frame);
    printf("Done\n");
}
