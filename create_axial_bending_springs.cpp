//#####################################################################
// Copyright 2019, Zhenglin Geng.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include <PhysBAM_Tools/Log/LOG.h>
#include <PhysBAM_Tools/Parsing/PARSE_ARGS.h>
#include <PhysBAM_Tools/Arrays_Computations/ARRAY_COPY.h>
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
#include <PhysBAM_Solids/PhysBAM_Deformables/Forces/AXIAL_BENDING_SPRINGS.h>
#include "GRID_TEST.h"
#include "FLAT_CLOTH_TEST.h"
#include <fstream>
#include <iostream>
#include <string>
using namespace PhysBAM;

typedef double T;
typedef VECTOR<T,3> TV;
typedef float RW;
void Write_Axial_Info(const std::string &tri_path,const std::string &out_dir)
{
    STREAM_TYPE stream_type((RW()));
    PARTICLES<TV>& particles=*new PARTICLES<TV>;
    TRIANGULATED_SURFACE<T> *surface=TRIANGULATED_SURFACE<T>::Create(particles);
    FILE_UTILITIES::Read_From_File(stream_type,tri_path,*surface);
    particles.Store_One_Over_Effective_Mass();
    ARRAYS_COMPUTATIONS::Fill(particles.mass,1.);
    ARRAYS_COMPUTATIONS::Fill(particles.one_over_effective_mass,1.);
    AXIAL_BENDING_SPRINGS<T> *axial_springs=Create_Axial_Bending_Springs(particles,surface->mesh,1e-4,1.,1.,false);
    std::string spring_particles_path=out_dir+"/axial_spring_particles.txt";
    std::cout<<"write to "<<spring_particles_path<<std::endl;
    std::ofstream fout_particles(spring_particles_path);
    std::string particle_weights_path=out_dir+"/axial_particle_weights.txt";
    std::cout<<"write to "<<particle_weights_path<<std::endl;
    std::ofstream fout_weights(particle_weights_path);
    fout_weights<<std::setprecision(100);
    ARRAY<VECTOR<int,4> > &spring_particles=axial_springs->spring_particles;
    for(int sp=1;sp<spring_particles.m;sp++){
        VECTOR<int,4> spv=spring_particles(sp);
        fout_particles<<spv(1)-1<<" "<<spv(2)-1<<" "<<spv(3)-1<<" "<<spv(4)-1<<std::endl;

        T axial_length;TV axial_direction;VECTOR<T,2> weights;T attached_edge_length;
        axial_springs->Axial_Vector(spv,axial_length,axial_direction,weights,attached_edge_length);
        fout_weights<<1-weights.x<<" "<<weights.x<<" "<<1-weights.y<<" "<<weights.y<<std::endl;
    }
}

int main(int argc,char *argv[])
{
    LOG::Initialize_Logging(false,false,1<<30,true,1);
    printf("\n");

    Initialize_Read_Write_General_Structures();

    PARSE_ARGS parse_args;
    parse_args.Add_String_Argument("-flat_cloth_path","","flat_cloth_path");
    parse_args.Add_String_Argument("-out_dir","","out_dir");
    parse_args.Parse(argc, argv);

    std::string flat_cloth_path=parse_args.Get_String_Value("-flat_cloth_path");
    std::string out_dir=parse_args.Get_String_Value("-out_dir");
    Write_Axial_Info(flat_cloth_path,out_dir);
}