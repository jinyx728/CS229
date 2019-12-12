//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "io_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>

template<class T>
void read_txt(std::string path,std::vector<T> &v)
{
    std::ifstream fin(path);
    if(!fin.good()){
        std::cout<<"cannot find "<<path<<std::endl;
    }
    while(true){
        pfloat t;fin>>t;
        if(!fin.good()) break;
        v.push_back((T)t);
    }
}

template void read_txt<pfloat>(std::string path,std::vector<pfloat> &v);
template void read_txt<idxint>(std::string path,std::vector<idxint> &v);

void read_obj(const std::string obj_path,std::vector<pfloat> &vts,std::vector<idxint> &fcs)
{
    std::ifstream fin(obj_path);
    while(true){
        std::string line;std::getline(fin,line);
        if(!fin.good())
            break;
        if(line.length()==0)
            continue;
        std::stringstream line_stream(line);
        std::string label;line_stream>>label;
        if(label=="v"){
            pfloat x,y,z;line_stream>>x>>y>>z;
            vts.push_back(x);vts.push_back(y);vts.push_back(z);
        }
        if(label=="f"){
            idxint x,y,z;line_stream>>x>>y>>z;
            fcs.push_back(x);fcs.push_back(y);fcs.push_back(z);
        }
    }
}

void write_obj(const std::string obj_path,const std::vector<pfloat> &vts,const std::vector<idxint> &fcs)
{
    std::ofstream fout(obj_path);
    idxint vi=0;
    while(vi<(int)vts.size()){
        fout<<"v "<<vts[vi]<<" "<<vts[vi+1]<<" "<<vts[vi+2]<<std::endl;
        vi+=3;
    }
    idxint fi=0;
    while(fi<(int)fcs.size()){
        fout<<"f "<<fcs[fi]+1<<" "<<fcs[fi+1]+1<<" "<<fcs[fi+2]+1<<std::endl;
        fi+=3;
    }
}