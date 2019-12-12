//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "spring_system.h"
#include <cuda.h>

using namespace cudaqs;
#define D 3

__device__ double norm(double* __restrict__ a,int n)
{
    double sum=0.0;
    for(int i=0;i<n;i++){
        sum+=a[i]*a[i];
    }
    return sqrt(sum);
}

__device__ double dot(double* __restrict__ a,double* __restrict__ b,int n)
{
    double sum=0;
    for(int i=0;i<n;i++){
        sum+=a[i]*b[i];
    }
    return sum;
}

__global__
void get_spring_data_kernel(const double* __restrict__ x,const int* __restrict__ edges,const double* __restrict__ l0,int n_vts,int n_edges,double* __restrict__ d,double* __restrict__ l,double* __restrict__ lhat,double* __restrict__ l0_over_l)
{
    int edge_i=threadIdx.x+blockIdx.x*blockDim.x;
    if(edge_i>=n_edges)
        return;
    int i0=edges[edge_i],i1=edges[n_edges+edge_i];
    double xi0[D],xi1[D],di[D],lhati[D];
    for(int j=0;j<D;j++){
        xi0[j]=x[j*n_vts+i0];
        xi1[j]=x[j*n_vts+i1];
        di[j]=xi1[j]-xi0[j];
    }
    double li=norm(di,D);
    double ratioi=l0[edge_i]/li;
    for(int j=0;j<D;j++){
        lhati[j]=di[j]/li;
    }
    for(int j=0;j<D;j++){
        d[j*n_edges+edge_i]=di[j];
        l[edge_i]=li;
        lhat[j*n_edges+edge_i]=lhati[j];
        l0_over_l[edge_i]=ratioi;
    }
}

__global__
void add_anchor_J_kernel(const double* __restrict__ x,const double* __restrict__ stiffen_anchor,const double* __restrict__ anchor,int n_vts,double* __restrict__ J)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n_vts)
        return;
    double k=stiffen_anchor[i];
    for(int j=0;j<D;j++){
        int p=j*n_vts+i;
        atomicAdd(&J[p],(x[p]-anchor[p])*k);
    }
}

__global__
void add_spring_J_kernel(const double* __restrict__ x,const int* __restrict__ edges,const double* __restrict__ l0,const double* __restrict__ k,const double* __restrict__ l,const double* __restrict__ lhat,int n_vts,int n_edges,double* __restrict__ J)
{
    int edge_i=threadIdx.x+blockIdx.x*blockDim.x;
    if(edge_i>=n_edges)
        return;
    int i0=edges[edge_i],i1=edges[n_edges+edge_i];
    double v=(l[edge_i]-l0[edge_i])*k[edge_i];
    for(int j=0;j<D;j++){
        double r=v*lhat[j*n_edges+edge_i];
        atomicAdd(&J[j*n_vts+i0],-r);
        atomicAdd(&J[j*n_vts+i1],r);
    }
}

__global__
void axial_gather_scatter(const double* __restrict__ x,const int* __restrict__ axial_i,const double* __restrict__ axial_w,const double* __restrict__ axial_k,int n_vts,int n_edges,double* __restrict__ out)
{
    int edge_i=threadIdx.x+blockIdx.x*blockDim.x;
    if(edge_i>=n_edges)
        return;
    int i[4];
    double w[4];
    for(int j=0;j<4;j++){
        i[j]=axial_i[j*n_edges+edge_i];
        w[j]=axial_w[j*n_edges+edge_i];
    }
    double ak=axial_k[edge_i];
    double xi[4][D],kr[D];
    for(int j=0;j<4;j++){
        for(int k=0;k<D;k++){
            xi[j][k]=x[k*n_vts+i[j]];
        }
    }
    for(int k=0;k<D;k++){
        kr[k]=(xi[0][k]*w[0]+xi[1][k]*w[1]-xi[2][k]*w[2]-xi[3][k]*w[3])*ak;
    }
    for(int k=0;k<D;k++){
        atomicAdd(&out[k*n_vts+i[0]],kr[k]*w[0]);
        atomicAdd(&out[k*n_vts+i[1]],kr[k]*w[1]);
        atomicAdd(&out[k*n_vts+i[2]],-kr[k]*w[2]);
        atomicAdd(&out[k*n_vts+i[3]],-kr[k]*w[3]);
    }
}

__global__
void add_anchor_Hu_kernel(const double* __restrict__ u,const double* __restrict__ stiffen_anchor,int n_vts,double* __restrict__ Hu)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n_vts)
        return;
    double k=stiffen_anchor[i];
    for(int j=0;j<D;j++){
        int p=j*n_vts+i;
        atomicAdd(&Hu[p],k*u[p]);
    }
}

__global__
void add_spring_Hu_kernel(const double* __restrict__ u,const int* __restrict__ edges,const double* __restrict__ k,const double* __restrict__ lhat,const double* __restrict__ l0_over_l,int n_vts,int n_edges,double* __restrict__ Hu)
{
    int edge_i=threadIdx.x+blockIdx.x*blockDim.x;
    if(edge_i>=n_edges)
        return;
    int i0=edges[edge_i],i1=edges[n_edges+edge_i];
    double du[D],lhati[D];
    for(int j=0;j<D;j++){
        du[j]=u[j*n_vts+i1]-u[j*n_vts+i0];
        lhati[j]=lhat[j*n_edges+edge_i];
    }
    double v=dot(du,lhati,D);
    double c=1-l0_over_l[edge_i];
    double ki=k[edge_i];
    if(c<0) c=0;
    for(int j=0;j<D;j++){
        double luj=v*lhati[j];
        double lvj=du[j]-luj;
        double rj=ki*(c*lvj+luj);
        atomicAdd(&Hu[j*n_vts+i0],-rj);
        atomicAdd(&Hu[j*n_vts+i1],rj);
    }
}

__global__
void get_vertex_norm(const double* __restrict__ x,int n_vts,double* __restrict__ n)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n_vts)
        return;
    // __shared__ 
    double xi[D];
    for(int k=0;k<D;k++){
        xi[k]=x[k*n_vts+i];
    }
    n[i]=norm(xi,D);
}

void SpringSystem::init(int n_vts_input,const SpringData &spring_data_input,const AxialData &axial_data_input)
{
    n_vts=n_vts_input;
    spring_data=spring_data_input;
    axial_data=axial_data_input;
}

void SpringSystem::get_data(const Tensor<double> &x,SystemData &opt_data) const
{
    //assert(x.size==opt_data.x.size);
    const int block_size=256;
    const int num_blocks_spring=(spring_data.n_edges+block_size-1)/block_size;
    opt_data.x=x;
    const cudaStream_t stream=opt_data.ctx_ptr->stream;
    // printf("x:%d,edges:%d,l0:%d,d:%d,l:%d,lhat:%d,l0_over_l:%d,n_edges:%d\n",(int)x.v.size(),(int)spring_data.edges.v.size(),(int)spring_data.l0.v.size(),(int)opt_data.d.v.size(),(int)opt_data.l.v.size(),(int)opt_data.lhat.v.size(),(int)opt_data.l0_over_l.v.size(),spring_data.n_edges);
    get_spring_data_kernel<<<num_blocks_spring,block_size,0,stream>>>(x.data(),spring_data.edges.data(),spring_data.l0.data(),n_vts,spring_data.n_edges,opt_data.d.data(),opt_data.l.data(),opt_data.lhat.data(),opt_data.l0_over_l.data());
}

void SpringSystem::get_J(const SystemData &opt_data,Tensor<double> &J) const
{
    const int block_size=256;
    const int num_blocks_anchor=(n_vts+block_size-1)/block_size;
    const int num_blocks_spring=(spring_data.n_edges+block_size-1)/block_size;
    const int num_blocks_axial=(axial_data.n_edges+block_size-1)/block_size;
    const cudaStream_t stream=opt_data.ctx_ptr->stream;
    J.set_zero();
    add_anchor_J_kernel<<<num_blocks_anchor,block_size,0,stream>>>(opt_data.x.data(),opt_data.stiffen_anchor.data(),opt_data.anchor.data(),n_vts,J.data());
    add_spring_J_kernel<<<num_blocks_spring,block_size,0,stream>>>(opt_data.x.data(),spring_data.edges.data(),spring_data.l0.data(),spring_data.k.data(),opt_data.l.data(),opt_data.lhat.data(),n_vts,spring_data.n_edges,J.data());
    axial_gather_scatter<<<num_blocks_axial,block_size,0,stream>>>(opt_data.x.data(),axial_data.i.data(),axial_data.w.data(),axial_data.k.data(),n_vts,axial_data.n_edges,J.data());
}

void SpringSystem::get_Hu(const SystemData &opt_data,const Tensor<double> &u,Tensor<double> &Hu) const
{
    const int block_size=256;
    const int num_blocks_anchor=(n_vts+block_size-1)/block_size;
    const int num_blocks_spring=(spring_data.n_edges+block_size-1)/block_size;
    const int num_blocks_axial=(axial_data.n_edges+block_size-1)/block_size;
    const cudaStream_t stream=opt_data.ctx_ptr->stream;
    Hu.set_zero();
    add_anchor_Hu_kernel<<<num_blocks_anchor,block_size,0,stream>>>(u.data(),opt_data.stiffen_anchor.data(),n_vts,Hu.data());
    add_spring_Hu_kernel<<<num_blocks_spring,block_size,0,stream>>>(u.data(),spring_data.edges.data(),spring_data.k.data(),opt_data.lhat.data(),opt_data.l0_over_l.data(),n_vts,spring_data.n_edges,Hu.data());
    axial_gather_scatter<<<num_blocks_axial,block_size,0,stream>>>(u.data(),axial_data.i.data(),axial_data.w.data(),axial_data.k.data(),n_vts,axial_data.n_edges,Hu.data());
}

void SpringSystem::mul(const SystemData &opt_data,const Tensor<double> &u,Tensor<double> &Hu) const
{
    get_Hu(opt_data,u,Hu);
}

double SpringSystem::inner(const Tensor<double> &x,const Tensor<double> &y) const
{
    return x.inner(y);
}

double SpringSystem::convergence_norm(const SystemData &opt_data,const Tensor<double> &r, Tensor<double> &n) const
{
    const int block_size=256;
    const int num_blocks=(n_vts+block_size-1)/block_size;
    const cudaStream_t stream=opt_data.ctx_ptr->stream;
    get_vertex_norm<<<block_size,num_blocks,0,stream>>>(r.data(),n_vts,n.data());
    return n.max()/opt_data.J_rms;
}

void SpringSystem::precondition(const SystemData &opt_data,const Tensor<double> &in_x,Tensor<double> &out_x) const
{
    out_x=in_x;
}