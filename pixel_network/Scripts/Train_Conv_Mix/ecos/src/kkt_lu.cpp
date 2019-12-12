#include "kkt_lu.h"
#include <math.h>
#include <vector>
#include <Eigen/SparseCore>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Dense>

typedef Eigen::SparseMatrix<pfloat,Eigen::ColMajor> SpaMat;

static Eigen::VectorXd Eigen_From_Raw(const pfloat *data,const int n)
{
    Eigen::VectorXd v(n);
    for(int i=0;i<n;i++){
        v[i]=data[i];
    }
    return v;
}

static void create_ccs_mat(spmat *X,SpaMat &L)
{
    L.resize(X->m,X->n);
    typedef Eigen::Triplet<double> Triplet;
    std::vector<Triplet> triplets;
    for(int coli=0;coli<X->n;coli++){
        for(int pi=X->jc[coli];pi<X->jc[coli+1];pi++){
            pfloat v=X->pr[pi];
            int row=X->ir[pi];
            triplets.push_back(Triplet(row,coli,v));
            if(row!=coli){
                triplets.push_back(Triplet(coli,row,v));
            }
        }
    }
    L.setFromTriplets(triplets.begin(),triplets.end());
    L.makeCompressed();
}

static void create_dense_mat(spmat *X,Eigen::MatrixXd &L)
{
    L.resize(X->m,X->n);
    for(int coli=0;coli<X->n;coli++){
        for(int pi=X->jc[coli];pi<X->jc[coli+1];pi++){
            pfloat v=X->pr[pi];
            int row=X->ir[pi];
            L(row,coli)=v;
            if(row!=coli){
                L(coli,row)=v;
            }
        }
    }
}


extern "C" idxint lu_solve(kkt* KKT, spmat* A, spmat* G, pfloat* Pb, pfloat* dx, pfloat* dy, pfloat* dz, idxint n, idxint p, idxint m, cone* C)
{
    idxint nK=n+p+m;
#if CONEMODE == 0
    nK += 2*C->nsoc;
#endif
    Eigen::VectorXd rhs=Eigen_From_Raw(Pb,nK);
    SpaMat PKP;
    create_ccs_mat(KKT->PKPt,PKP);
    Eigen::SparseLU<SpaMat, Eigen::AMDOrdering<int> > solver;
    // Eigen::SparseLU<SpaMat, Eigen::COLAMDOrdering<int> > solver;
    // Eigen::SparseLU<SpaMat, Eigen::NaturalOrdering<int> > solver;
    // printf("analyzePattern\n");
    solver.analyzePattern(PKP);
    // printf("factorize(natural)\n");
    solver.factorize(PKP);
    // printf("solve\n");
    Eigen::VectorXd Px=solver.solve(rhs);
    Eigen::VectorXd e=rhs-PKP*Px;
    // printf("lu_solve,min:%.17g,max:%.17g\n",e.minCoeff(),e.maxCoeff());
    unstretch(n,p,C,KKT->Pinv,Px.data(),dx,dy,dz);
}

extern "C" idxint dense_lu_solve(kkt* KKT, spmat* A, spmat* G, pfloat* Pb, pfloat* dx, pfloat* dy, pfloat* dz, idxint n, idxint p, idxint m, cone* C)
{
    idxint nK=n+p+m;
#if CONEMODE == 0
    nK += 2*C->nsoc;
#endif
    Eigen::VectorXd rhs=Eigen_From_Raw(Pb,nK);
    Eigen::MatrixXd PKP;
    printf("create_dense_mat,nK:%d\n",(int)nK);
    create_dense_mat(KKT->PKPt,PKP);
    printf("init solver\n");
    Eigen::PartialPivLU<Eigen::MatrixXd> solver;
    Eigen::VectorXd Px=solver.solve(rhs);
    Eigen::VectorXd e=rhs-PKP*Px;
    printf("dense_lu_solve,min:%.17g,max:%.17g\n",e.minCoeff(),e.maxCoeff());
    unstretch(n,p,C,KKT->Pinv,Px.data(),dx,dy,dz);
}

extern "C" idxint bcg_solve(kkt* KKT, spmat* A, spmat* G, pfloat* Pb, pfloat* dx, pfloat* dy, pfloat* dz, idxint n, idxint p, idxint m, cone* C)
{
    idxint nK=n+p+m;
#if CONEMODE == 0
    nK += 2*C->nsoc;
#endif
    Eigen::VectorXd rhs=Eigen_From_Raw(Pb,nK);
    SpaMat PKP;
    create_ccs_mat(KKT->PKPt,PKP);
    Eigen::BiCGSTAB<SpaMat,Eigen::IncompleteLUT<double> > solver;
    solver.compute(PKP);
    Eigen::VectorXd Px=solver.solve(rhs);
    Eigen::VectorXd e=rhs-PKP*Px;
    printf("bcg_solve,min:%.17g,max:%.17g\n",e.minCoeff(),e.maxCoeff());
    unstretch(n,p,C,KKT->Pinv,Px.data(),dx,dy,dz);
}

extern "C" idxint qr_solve(kkt* KKT, spmat* A, spmat* G, pfloat* Pb, pfloat* dx, pfloat* dy, pfloat* dz, idxint n, idxint p, idxint m, cone* C)
{
    idxint nK=n+p+m;
#if CONEMODE == 0
    nK += 2*C->nsoc;
#endif
    Eigen::VectorXd rhs=Eigen_From_Raw(Pb,nK);
    SpaMat PKP;
    create_ccs_mat(KKT->PKPt,PKP);
    Eigen::SparseQR<SpaMat, Eigen::AMDOrdering<int> > solver;
    printf("analyzePattern\n");
    solver.analyzePattern(PKP);
    printf("factorize\n");
    solver.factorize(PKP);
    printf("solve\n");
    Eigen::VectorXd Px=solver.solve(rhs);
    Eigen::VectorXd e=rhs-PKP*Px;
    printf("qr_solve,min:%.17g,max:%.17g\n",e.minCoeff(),e.maxCoeff());
    unstretch(n,p,C,KKT->Pinv,Px.data(),dx,dy,dz);
}


extern "C" void check_consistency(pfloat *dx1_p, pfloat *dy1_p, pfloat *dz1_p, pfloat *dx2_p, pfloat *dy2_p, pfloat *dz2_p, idxint n, idxint p, idxint m)
{
    Eigen::VectorXd dx1=Eigen_From_Raw(dx1_p,n),dy1=Eigen_From_Raw(dy1_p,p),dz1=Eigen_From_Raw(dz1_p,m);
    Eigen::VectorXd dx2=Eigen_From_Raw(dx2_p,n),dy2=Eigen_From_Raw(dy2_p,p),dz2=Eigen_From_Raw(dz2_p,m);
    Eigen::VectorXd ex=(dx2-dx1).array().abs().matrix();
    Eigen::VectorXd ey=(dy2-dy1).array().abs().matrix();
    Eigen::VectorXd ez=(dz2-dz1).array().abs().matrix();
    printf("check:ex:%.17g,ey:%.17g,ez:%.17g\n",ex.maxCoeff(),ey.maxCoeff(),ez.maxCoeff());
}