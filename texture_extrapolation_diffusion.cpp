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
                                               ARRAY<T>& in_dx, ARRAY<T>& in_dy, ARRAY<VECTOR<T,2> >& vt, std::string out_folder, std::string frame)
{
    typedef VECTOR<T,3> TV;

    // Signed distance field
    T stopping_distance = 0;
    FAST_MARCHING_METHOD_SURFACE_MESH<TV> fmm_x(*triangulated_surface, *triangulated_surface->vertex_normals);
    fmm_x.Fast_Marching_Method(phi,stopping_distance,&seed_indices);

    //Get back indices to set as Dirichlet
    /*
    std::string back_vertices_filename="/data/jwu/PhysBAM/Private_Projects/cloth_on_virtual_body/Learning/shared_data/back_vertices.txt";
    std::ifstream back_vertices_file(back_vertices_filename);
    T front_id;
    ARRAY<int> back_indices;
    while (back_vertices_file >> front_id) {
        const int i = (int)front_id + 1; //Zero indexing
        back_indices.Append(i);
    }
    std::cout << "Num back indices: " << back_indices.Size() << std::endl;
    */

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
    printf("Done with laplace solve\n");

    /*
    //Reset back vertices
    for (int i = 1; i <= back_indices.Size(); i++) {
        in_dx(back_indices(i)) = 0;
        in_dy(back_indices(i)) = 0;
    }*/

    std::string output_phi_filename = out_folder + "/diffusion_phi.txt";
    std::string output_vt_filename = out_folder + "/diffusion_vt.txt";
    std::string output_disp_filename = out_folder + "/displace_" + frame + ".txt";
   
    // Write output phi
    std::ostream* output=FILE_UTILITIES::Safe_Open_Output(output_phi_filename,false,false);
    for (int i = 1; i <= phi.Size(); i++)
        *output << phi(i) << "\n";
    delete output;

    // Write output texture coordinates
    output=FILE_UTILITIES::Safe_Open_Output(output_vt_filename,false,false);
    for (int i = 1; i <= in_dx.Size(); i++)
        *output << "vt " << vt(i)(1) + in_dx(i) << " " << vt(i)(2) + in_dy(i) << "\n";
    delete output;

    // Write output texture coordinates
    output=FILE_UTILITIES::Safe_Open_Output(output_vt_filename,false,false);
    for (int i = 1; i <= in_dx.Size(); i++)
        *output << "vt " << vt(i)(1) + in_dx(i) << " " << vt(i)(2) + in_dy(i) << "\n";
    delete output;

    // Write output texture coordinates
    output=FILE_UTILITIES::Safe_Open_Output(output_disp_filename,false,false);
    for (int i = 1; i <= in_dx.Size(); i++)
        *output << in_dx(i) << " " << in_dy(i) << "\n";
    delete output;
}

template<class T,class RW> void Flood_Fill(std::string input_mesh_filename, std::string input_vt_filename, std::string input_disp_filename, std::string out_folder, std::string frame)
{
    typedef VECTOR<T,3> TV; 

    //Read in cloth mesh
    TRIANGULATED_SURFACE<T>* triangulated_surface=0;
    FILE_UTILITIES::Create_From_File<RW>(input_mesh_filename,triangulated_surface);
    triangulated_surface->Update_Vertex_Normals();
    triangulated_surface->mesh.Initialize_Neighbor_Nodes();
    triangulated_surface->mesh.Initialize_Incident_Elements();
    
    //Read in ground truth texture coordinates
    std::ifstream vt_file(input_vt_filename);
    std::string label;
    T a, b;
    ARRAY<VECTOR<T,2> > vt;
    while (vt_file >> label >> a >> b){
        vt.Append(VECTOR<T,2>(a,b));
    }

    //Read in texture coordinate displacements after ray-intersection method
    std::ifstream disp_file(input_disp_filename);
    //T a, b;
    int count = 0;
    int idx = 1;
    ARRAY<T> in_dx;
    ARRAY<T> in_dy;
    ARRAY<T> phi; //Hard coded for now...
    // T prev_phi = 1;
    while (disp_file >> a >> b){
        in_dx.Append(a);
        in_dy.Append(b);
        //Set starting phi values
        if (a==0.0 && b ==0.0) {
            count += 1;
            phi.Append(1);
        }
        else {
            phi.Append(-1);
        }
        // prev_phi = phi(idx);
        idx += 1;
    }
    std::cout << "Cloth tc: " << in_dx.Size() << ", " << in_dy.Size() << std::endl;
    std::cout << "Phi: " << phi.Size() << std::endl;
    std::cout << "Num missing: " << count << std::endl;

    //Set final seed indices
    ARRAY<int> seed_indices;
    SEGMENT_MESH& segment_mesh = triangulated_surface->Get_Segment_Mesh();
    ARRAY<T> is_seed(triangulated_surface->particles.array_collection->Size());
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

    Fast_Marching<T,RW>(triangulated_surface, phi, seed_indices, in_dx, in_dy, vt, out_folder, frame);
}

int main(int argc,char *argv[])
{
    LOG::Initialize_Logging(false,false,1<<30,true,1);
    printf("\n");

    PROCESS_UTILITIES::Set_Floating_Point_Exception_Handling(true);
    Initialize_Read_Write_General_Structures();

    PARSE_ARGS parse_args;
    parse_args.Set_Extra_Arguments(1,"<input_mesh>","<input mesh> mesh to extrapolate on");
    parse_args.Set_Extra_Arguments(2,"<texture coords>","<texture coords> texture coordinates (pure)");
    parse_args.Set_Extra_Arguments(3,"<displacements>","<displacements> displacement values for tex coords (pre-flood fill)");
    parse_args.Set_Extra_Arguments(4,"<output folder>","<output folder> output folder for extrapolation");
    parse_args.Set_Extra_Arguments(5,"<frame>","<frame> pose frame number");
    parse_args.Parse(argc, argv);

    std::string input_mesh_filename=parse_args.Extra_Arg(1);
    std::string input_vt_filename=parse_args.Extra_Arg(2);
    std::string input_disp_filename=parse_args.Extra_Arg(3);
    std::string out_folder=parse_args.Extra_Arg(4);
    std::string frame = parse_args.Extra_Arg(5);
 
    Flood_Fill<float,float>(input_mesh_filename, input_vt_filename, input_disp_filename, out_folder, frame);

    //2) Flat Cloth Test
    /*parse_args.Set_Extra_Arguments(1,"<texture coords>","<texture coords> texture coordinates (pure)");
    parse_args.Set_Extra_Arguments(2,"<displacements>","<displacements> displacement values for tex coords (pre-flood fill)");
    parse_args.Set_Extra_Arguments(3,"<output folder>","<output folder> output folder for extrapolation");
    parse_args.Parse(argc, argv);

    std::string input_vt_filename=parse_args.Extra_Arg(1);
    std::string input_disp_filename=parse_args.Extra_Arg(2);
    std::string output_folder = parse_args.Extra_Arg(3);
    std::cout << output_folder + "\n";

    FLAT_CLOTH_TEST<float>* test = new FLAT_CLOTH_TEST<float>(input_vt_filename, input_disp_filename, output_folder);
    test->Extrapolate();
    test->Write_Outputs();
    */

    //1) 2D Grid Test
    /*
    parse_args.Add_Option_Argument("-circle","use circle interface");
    parse_args.Set_Extra_Arguments(1,"<tri file>","<tri file> tri file to be flood filled");
    parse_args.Set_Extra_Arguments(2,"<output folder>","<output folder> output folder for extrapolation");
    parse_args.Parse(argc, argv);

    std::string input_mesh_filename = parse_args.Extra_Arg(1);
    std::string output_folder = parse_args.Extra_Arg(2);
    std::cout << output_folder + "\n";

    GRID_TEST<float>* test = new GRID_TEST<float>(input_mesh_filename, output_folder, parse_args.Get_Option_Value("-circle"));

    //Read in values to be extrapolated
    std::string input_disp_filename="/data/jwu/PhysBAM/Private_Projects/cloth_texture/tests/grid/grid_vt.txt";
    std::ifstream disp_file(input_disp_filename);
    float a;
    ARRAY<float> u_input(test->phi_final.Size());
    int id = 1;
    while (disp_file >> a){
        if (test->phi_final(id) <= 0)
            u_input(id) = a;
        id += 1;
    }

    test->Extrapolate(u_input);
    test->Write_Outputs(u_input);
    */

    printf("Done\n");
}
