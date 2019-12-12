#ifndef __KKT_LU_H_
#define __KKT_LU_H_


#ifdef __cplusplus
extern "C" {
#endif

#include "kkt.h"
#include "ldl.h"
#include "splamm.h"
#include "ecos.h"
#include "cone.h"
idxint lu_solve(kkt* KKT, spmat* A, spmat* G, pfloat* Pb, pfloat* dx, pfloat* dy, pfloat* dz, idxint n, idxint p, idxint m, cone* C);
idxint dense_lu_solve(kkt* KKT, spmat* A, spmat* G, pfloat* Pb, pfloat* dx, pfloat* dy, pfloat* dz, idxint n, idxint p, idxint m, cone* C);
idxint bcg_solve(kkt* KKT, spmat* A, spmat* G, pfloat* Pb, pfloat* dx, pfloat* dy, pfloat* dz, idxint n, idxint p, idxint m, cone* C);
idxint qr_solve(kkt* KKT, spmat* A, spmat* G, pfloat* Pb, pfloat* dx, pfloat* dy, pfloat* dz, idxint n, idxint p, idxint m, cone* C);
void check_consistency(pfloat *dx1, pfloat *dy1, pfloat *dz1, pfloat *dx2, pfloat *dy2, pfloat *dz2, idxint n, idxint p, idxint m);
#ifdef __cplusplus
}
#endif

#endif