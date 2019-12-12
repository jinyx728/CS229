//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "io_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace cudaqs;

template<class T>
void cudaqs::read_txt(std::string path,std::vector<T> &v)
{
    v.clear();
    std::ifstream fin(path);
    if(!fin.good()){
        std::cout<<"cannot find "<<path<<std::endl;
    }
    while(true){
        double t;fin>>t;
        if(!fin.good()) break;
        v.push_back((T)t);
    }
}

template<class T>
void cudaqs::write_txt(std::string path,const std::vector<T> &v)
{
    std::ofstream fout(path);
    if(!fout.good()){
        std::cout<<"cannot write to "<<path<<std::endl;
    }
    fout<<std::setprecision(100);
    for(const auto t:v){
        fout<<t<<std::endl;
    }
}

template void cudaqs::read_txt<double>(std::string path,std::vector<double> &v);
template void cudaqs::read_txt<int>(std::string path,std::vector<int> &v);
template void cudaqs::write_txt<double>(std::string path,const std::vector<double> &v);

void cudaqs::read_obj(const std::string obj_path,std::vector<double> &vts,std::vector<int> &fcs)
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
            double x,y,z;line_stream>>x>>y>>z;
            vts.push_back(x);vts.push_back(y);vts.push_back(z);
        }
        if(label=="f"){
            int x,y,z;line_stream>>x>>y>>z;
            fcs.push_back(x);fcs.push_back(y);fcs.push_back(z);
        }
    }
}

void cudaqs::write_obj(const std::string obj_path,const std::vector<double> &vts,const std::vector<int> &fcs)
{
    std::ofstream fout(obj_path);
    int vi=0;
    while(vi<(int)vts.size()){
        fout<<"v "<<vts[vi]<<" "<<vts[vi+1]<<" "<<vts[vi+2]<<std::endl;
        vi+=3;
    }
    int fi=0;
    while(fi<(int)fcs.size()){
        fout<<"f "<<fcs[fi]+1<<" "<<fcs[fi+1]+1<<" "<<fcs[fi+2]+1<<std::endl;
        fi+=3;
    }
}

template<class T>
void cudaqs::transpose(const std::vector<T> &in_v,std::vector<T> &out_v,std::vector<int> &size)
{
    out_v.resize(in_v.size());
    for(int i0=0;i0<size[0];i0++){
        for(int i1=0;i1<size[1];i1++){
            out_v[i1*size[0]+i0]=in_v[i0*size[1]+i1];
        }
    }
}

SpringData cudaqs::load_spring_data(std::string data_dir)
{
    SpringData spring_data;

    std::string edge_path=data_dir+"/edges.txt";
    std::vector<int> edges_v;read_txt(edge_path,edges_v);
    spring_data.n_edges=(int)edges_v.size()/2;
    std::vector<int> edges_vt=edges_v;
    std::vector<int> size={spring_data.n_edges,2};
    transpose<int>(edges_v,edges_vt,size);
    spring_data.edges.v=edges_vt;spring_data.edges.size={2,spring_data.n_edges};

    std::string l0_path=data_dir+"/l0.txt";
    std::vector<double> l0_v;read_txt<double>(l0_path,l0_v);
    spring_data.l0.v=l0_v;spring_data.l0.size={(int)l0_v.size()};

    std::string k_path=data_dir+"/k.txt";
    std::vector<double> k_v;read_txt(k_path,k_v);
    spring_data.k.v=k_v;spring_data.k.size={(int)k_v.size()};

    return spring_data;
}

AxialData cudaqs::load_axial_data(std::string data_dir)
{
    AxialData axial_data;

    std::string i_path=data_dir+"/axial_i.txt";
    std::vector<int> i_v;read_txt(i_path,i_v);
    axial_data.n_edges=i_v.size()/4;
    std::vector<int> i_vt;
    std::vector<int> size={axial_data.n_edges,4};
    transpose<int>(i_v,i_vt,size);
    axial_data.i.v=i_vt;axial_data.i.size={4,axial_data.n_edges};

    std::string w_path=data_dir+"/axial_w.txt";
    std::vector<double> w_v;read_txt(w_path,w_v);
    std::vector<double> w_vt;
    transpose<double>(w_v,w_vt,size);
    axial_data.w.v=w_vt;axial_data.w.size={4,axial_data.n_edges};

    std::string k_path=data_dir+"/axial_k.txt";
    std::vector<double> k_v;read_txt(k_path,k_v);
    axial_data.k.v=k_v;axial_data.k.size={axial_data.n_edges};

    return axial_data;
}

void cudaqs::load_anchor_data(std::string data_dir,Tensor<double> &anchor,Tensor<double> &stiffen_anchor)
{
    std::string anchor_path=data_dir+"/anchor.txt";
    std::vector<double> anchor_v;read_txt(anchor_path,anchor_v);
    int n_vts=anchor_v.size()/3;
    std::vector<double> anchor_vt;
    std::vector<int> size={n_vts,3};
    transpose<double>(anchor_v,anchor_vt,size);
    anchor.v=anchor_vt;anchor.size={3,n_vts};

    std::string stiffen_anchor_path=data_dir+"/stiffen_anchor.txt";
    std::vector<double> stiffen_anchor_v;read_txt(stiffen_anchor_path,stiffen_anchor_v);
    stiffen_anchor.v=stiffen_anchor_v;stiffen_anchor.size={n_vts};
}

void cudaqs::read_txt_and_transpose(std::string path,Tensor<double> &t,std::vector<int> size)
{
    std::vector<double> v;read_txt<double>(path,v);
    std::vector<double> vt;transpose(v,vt,size);
    t.v=vt;t.size={size[1],size[0]};
}