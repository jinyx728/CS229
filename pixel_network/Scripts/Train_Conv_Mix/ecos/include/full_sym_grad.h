#ifndef __FULL_SYM_GRAD_H__
#define __FULL_SYM_GRAD_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "glblopts.h"
#include "spla.h"
#include "cone.h"
#include "kkt.h"

void full_sym_grad_update(spmat* PKP,idxint *P,cone *C,pfloat *s,pfloat *z);
void full_sym_grad_solve();
void full_sym_grad_unstretch();

#ifdef __cplusplus
}
#endif

#endif