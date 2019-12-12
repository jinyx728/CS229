//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once

/* DATA TYPES ---------------------------------------------------------- */
#include <float.h>
#include <math.h>
#include <stdio.h>

typedef double pfloat;              /* for numerical values  */
// typedef __float128 pfloat;              /* for numerical values  */
typedef int idxint;


#define ACOS_INFINITY   (DBL_MAX + DBL_MAX)
#define ACOS_NAN        (ACOS_INFINITY - ACOS_INFINITY)

#define PRINTTEXT printf
#define MALLOC malloc
#define FREE free
