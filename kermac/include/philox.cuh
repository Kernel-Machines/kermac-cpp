/* Copyright (c) 2025 Kernel Machines, Charles Durham
*/

/* Copyright 2010-2014 NVIDIA Corporation.  All rights reserved.
 *                   
 * NOTICE TO LICENSEE:
 *
 * The source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * The Licensed Deliverables contained herein are PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and are being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
/*
   Copyright 2010-2011, D. E. Shaw Research.
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions, and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions, and the following disclaimer in the
 documentation and/or other materials provided with the distribution.

 * Neither the name of D. E. Shaw Research nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

// Modified from Nvidia's philox to match behavior of their philox with zero state initialization.

/* LIMITATIONS

* State will not roll over to another 4 byte after it exceeds 2^32, it will repeat
* Seed is necessary as an input param
* Assumes that per thread offset is always going to be the unique thread id of the launch

*/

#include <stdint.h>

#define PHILOX_W32_0   (0x9E3779B9)
#define PHILOX_W32_1   (0xBB67AE85)
#define PHILOX_M4x32_0 (0xD2511F53)
#define PHILOX_M4x32_1 (0xCD9E8D57)
#define CURAND_2POW32_INV (2.3283064e-10f)
#define CURAND_2POW32_INV_2PI (2.3283064e-10f * 6.2831855f)

#define CURAND_PI_DOUBLE  (3.1415926535897932)
#define CURAND_2POW53_INV_DOUBLE (1.1102230246251565e-16)

#define QUALIFIERS static __forceinline__ __device__

// Finds the thread that is the furthest along in burning the rng
// generator and makes sure that value is the one that is placed
// in max_state_value
QUALIFIERS
void 
philox_put_state(uint32_t state, uint32_t *max_state_value) {
    int lane_id = threadIdx.x % 32;

    uint32_t warp_max;
#if __CUDA_ARCH__ >= 800
    warp_max = __reduce_max_sync(0xFFFFFFFF, state);
   
#else
    warp_max = state;
    for (int offset = 16; offset > 0; offset /= 2) {
        uint32_t other = __shfl_down_sync(0xFFFFFFFF, warp_max, offset);
        warp_max = max(warp_max, other);
    }
#endif
    if (lane_id == 0) {
        atomicMax(max_state_value, warp_max);
    }
}

QUALIFIERS
uint4 
philox4(
    uint32_t thread_offset,
    uint32_t seed,
    uint32_t *iter_count
) {
    uint4 ctr = {*iter_count, 0, thread_offset, 0};
    uint32_t k0 = seed;
    uint32_t k1 = 0;
    // #pragma unroll 1
    for (int i = 0; i < 10; i++) {
        uint4 temp_ctr = {
            __umulhi(PHILOX_M4x32_1, ctr.z) ^ ctr.y ^ k0,
            PHILOX_M4x32_1 * ctr.z,
            __umulhi(PHILOX_M4x32_0, ctr.x) ^ ctr.w ^ k1,
            PHILOX_M4x32_0 * ctr.x
        };
        ctr = temp_ctr;
        k0 += PHILOX_W32_0;
        k1 += PHILOX_W32_1;
    }
    (*iter_count)++;
    return ctr;
}


QUALIFIERS 
float2 
_philox_box_muller(unsigned int x, unsigned int y)
{
    float u = x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
    float v = y * CURAND_2POW32_INV_2PI + (CURAND_2POW32_INV_2PI/2.0f);
    float s = sqrtf(-2.0f * logf(u));

    return (float2) {
        .x = __sinf(v) * s,
        .y = __cosf(v) * s
    };
}

QUALIFIERS 
double2
_philox_box_muller_double(
    unsigned int x0, 
    unsigned int x1,
    unsigned int y0, 
    unsigned int y1
) {
    unsigned long long zx = (unsigned long long)x0 ^
        ((unsigned long long)x1 << (53 - 32));
    double u = zx * CURAND_2POW53_INV_DOUBLE + (CURAND_2POW53_INV_DOUBLE/2.0);
    unsigned long long zy = (unsigned long long)y0 ^
        ((unsigned long long)y1 << (53 - 32));
    double v = zy * (CURAND_2POW53_INV_DOUBLE*2.0) + CURAND_2POW53_INV_DOUBLE;
    double s = sqrt(-2.0 * log(u));

    return (double2) {
        .x = sin(v*CURAND_PI_DOUBLE) * s,
        .y = cos(v*CURAND_PI_DOUBLE) * s
    };
}

QUALIFIERS
float4
philox_uniform4(
    uint32_t thread_offset,
    uint32_t seed,
    uint32_t *iter_count
) {
    uint4 xyzw = philox4(thread_offset, seed, iter_count);

    return (float4) {
        .x = xyzw.x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f),
        .y = xyzw.y * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f),
        .z = xyzw.z * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f),
        .w = xyzw.w * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f),
    };
}

QUALIFIERS
float4
philox_normal4(
    uint32_t thread_offset,
    uint32_t seed,
    uint32_t *iter_count
) {
    uint4 xyzw = philox4(thread_offset, seed, iter_count);

    float2 _result_x = _philox_box_muller(xyzw.x, xyzw.y);
    float2 _result_y = _philox_box_muller(xyzw.z, xyzw.w);

    return (float4) {
        .x = _result_x.x,
        .y = _result_x.y,
        .z = _result_y.x,
        .w = _result_y.y
    };
}

QUALIFIERS 
double4 
philox_normal4_double(
    uint32_t thread_offset,
    uint32_t seed,
    uint32_t *iter_count
) {
    uint4 _x;
    uint4 _y;
    double4 result;

    _x = philox4(thread_offset, seed, iter_count);
    _y = philox4(thread_offset, seed, iter_count);
    double2 v1 = _philox_box_muller_double(_x.x, _x.y, _x.z, _x.w);
    double2 v2 = _philox_box_muller_double(_y.x, _y.y, _y.z, _y.w);
    result.x = v1.x;
    result.y = v1.y;
    result.z = v2.x;
    result.w = v2.y;

    return result;
}

QUALIFIERS 
double 
_curand_uniform_double_hq(
    unsigned int x, 
    unsigned int y
) {
    unsigned long long z = (unsigned long long)x ^
        ((unsigned long long)y << (53 - 32));
    return z * CURAND_2POW53_INV_DOUBLE + (CURAND_2POW53_INV_DOUBLE/2.0);
}

QUALIFIERS 
double4 
philox_uniform4_double(
    uint32_t thread_offset,
    uint32_t seed,
    uint32_t *iter_count
) {
    uint4 _x;
    uint4 _y;
    double4 result;

    _x = philox4(thread_offset, seed, iter_count);
    _y = philox4(thread_offset, seed, iter_count);
    result.x = _curand_uniform_double_hq(_x.x,_x.y);
    result.y = _curand_uniform_double_hq(_x.z,_x.w);
    result.z = _curand_uniform_double_hq(_y.x,_y.y);
    result.w = _curand_uniform_double_hq(_y.z,_y.w);
    return result;
}


QUALIFIERS
float
_philox_uniform(
    uint32_t thread_offset,
    uint32_t seed,
    uint32_t *iter_count,
    float4 *rng_reservoir_ref,
    int *rng_used_ref
) {
    int rng_used = *rng_used_ref;
    float4 rng_reservoir = *rng_reservoir_ref;
    if (rng_used == 4) {
        rng_reservoir = philox_uniform4(
            thread_offset, seed, iter_count
        );
        rng_used = 0;
    }
    float rng_val;
    switch(rng_used++) {
        case 0: rng_val = rng_reservoir.x; break;
        case 1: rng_val = rng_reservoir.y; break;
        case 2: rng_val = rng_reservoir.z; break;
        case 3: rng_val = rng_reservoir.w; break;
    }

    *rng_reservoir_ref = rng_reservoir;
    *rng_used_ref = rng_used;

    return rng_val;
}

QUALIFIERS
uint4
_philox_poisson_knuth4(
    uint32_t thread_offset,
    uint32_t seed,
    uint32_t *iter_count,
    float lambda
) {
    uint4 k = {0,0,0,0};
    float exp_lambda = expf(lambda);
    float4 p = {exp_lambda, exp_lambda, exp_lambda, exp_lambda};

    float4 rng_reservoir;
    int rng_used = 4;

    do {
        k.x++;
        p.x *= _philox_uniform(thread_offset, seed, iter_count, &rng_reservoir, &rng_used);
    } while (p.x > 1.0);

    do {
        k.y++;
        p.y *= _philox_uniform(thread_offset, seed, iter_count, &rng_reservoir, &rng_used);
    } while (p.y > 1.0);

    do {
        k.z++;
        p.z *= _philox_uniform(thread_offset, seed, iter_count, &rng_reservoir, &rng_used);
    } while (p.z > 1.0);

    do {
        k.w++;
        p.w *= _philox_uniform(thread_offset, seed, iter_count, &rng_reservoir, &rng_used);
    } while (p.w > 1.0);

    k.x--;
    k.y--;
    k.z--;
    k.w--;
    return k;
}

QUALIFIERS
float
_cr_rsqrt(
    float a
) {
    asm ("rsqrt.approx.f32.ftz %0, %1;" : "=f"(a) : "f"(a));
    return a;
}

QUALIFIERS
float
_cr_exp(
    float a
) {
    a = a * 1.4426950408889634074;
    asm ("ex2.approx.f32.ftz %0, %1;" : "=f"(a) : "f"(a));
    return a;
}

QUALIFIERS 
float
_cr_log(
    float a
) {
    asm ("lg2.approx.f32.ftz %0, %1;" : "=f"(a) : "f"(a));
    a = a * 0.69314718055994530942;
    return a;
}

QUALIFIERS
float
_cr_rcp(
    float a
) {
    asm ("rcp.approx.f32.ftz %0, %1;" : "=f"(a) : "f"(a));
    return a;
}

/* Computes regularized gamma function:  gammainc(a,x)/gamma(a) */
QUALIFIERS
float
_cr_pgammainc(
    float a, 
    float x
) {
    float t, alpha, beta;

    /* First level parametrization constants */
    float ma1 = 1.43248035075540910f,
          ma2 = 0.12400979329415655f,
          ma3 = 0.00025361074907033f,
          mb1 = 0.21096734870196546f,
          mb2 = 1.97381164089999420f,
          mb3 = 0.94201734077887530f;

    /* Second level parametrization constants (depends only on a) */

    alpha = _cr_rsqrt(a - ma2);
    alpha = ma1 * alpha + ma3;
    beta = _cr_rsqrt(a - mb2);
    beta = mb1 * beta + mb3;

    /* Final approximation (depends on a and x) */

    t = a - x;
    t = alpha * t - beta;
    t = 1.0f + _cr_exp(t);
    t = t * t;
    t = _cr_rcp(t);

    /* Negative a,x or a,x=NAN requires special handling */
    //t = !(x > 0 && a >= 0) ? 0.0 : t;

    return t;
}

QUALIFIERS
float
_cr_pgammaincinv(
    float a, 
    float y
) {
    float t, alpha, beta;

    /* First level parametrization constants */

    float ma1 = 1.43248035075540910f,
          ma2 = 0.12400979329415655f,
          ma3 = 0.00025361074907033f,
          mb1 = 0.21096734870196546f,
          mb2 = 1.97381164089999420f,
          mb3 = 0.94201734077887530f;

    /* Second level parametrization constants (depends only on a) */

    alpha = _cr_rsqrt(a - ma2);
    alpha = ma1 * alpha + ma3;
    beta = _cr_rsqrt(a - mb2);
    beta = mb1 * beta + mb3;

    /* Final approximation (depends on a and y) */

    t = _cr_rsqrt(y) - 1.0f;
    t = _cr_log(t);
    t = beta + t;
    t = - t * _cr_rcp(alpha) + a;
    /* Negative a,x or a,x=NAN requires special handling */
    //t = !(y > 0 && a >= 0) ? 0.0 : t;
    return t;
}

static __constant__ double _cr_lgamma_table [] = {
    0.000000000000000000e-1,
    0.000000000000000000e-1,
    6.931471805599453094e-1,
    1.791759469228055001e0,
    3.178053830347945620e0,
    4.787491742782045994e0,
    6.579251212010100995e0,
    8.525161361065414300e0,
    1.060460290274525023e1
};

QUALIFIERS 
double 
_cr_lgamma_integer(
    int a
) {
    double s;
    double t;
    double fa = fabs((float)a);
    double sum;

    if (a > 8) {
        /* Stirling approximation; coefficients from Hart et al, "Computer
         * Approximations", Wiley 1968. Approximation 5404.
         */
        s = 1.0 / fa;
        t = s * s;
        sum =          -0.1633436431e-2;
        sum = sum * t + 0.83645878922e-3;
        sum = sum * t - 0.5951896861197e-3;
        sum = sum * t + 0.793650576493454e-3;
        sum = sum * t - 0.277777777735865004e-2;
        sum = sum * t + 0.833333333333331018375e-1;
        sum = sum * s + 0.918938533204672;
        s = 0.5 * log (fa);
        t = fa - 0.5;
        s = s * t;
        t = s - fa;
        s = s + sum;
        t = t + s;
        return t;
    } else {
        return _cr_lgamma_table [(int) fa-1];
    }
}

/* Rejection Method for Poisson distribution based on gammainc approximation */
QUALIFIERS
uint4 
_philox_poisson_gammainc4(
    uint32_t thread_offset,
    uint32_t seed,
    uint32_t *iter_count, 
    float lambda
){
    float4 rng_reservoir;
    int rng_used = 4;
    uint4 result;
    float y, x, t, z,v;
    float logl = _cr_log(lambda);
    while (true) {
        y = _philox_uniform(thread_offset, seed, iter_count, &rng_reservoir, &rng_used);
        x = _cr_pgammaincinv (lambda, y);
        x = floorf (x);
        z = _philox_uniform(thread_offset, seed, iter_count, &rng_reservoir, &rng_used);
        v = (_cr_pgammainc(lambda, x + 1.0f) - _cr_pgammainc(lambda, x)) * 1.3f;
        z = z*v;
        t = (float)_cr_exp(-lambda + x * logl - (float)_cr_lgamma_integer((int)(1.0f + x)));
        if ((z < t) && (v>=1e-20))
            break;
    }
    result.x = (unsigned int)x;

    while (true) {
        y = _philox_uniform(thread_offset, seed, iter_count, &rng_reservoir, &rng_used);
        x = _cr_pgammaincinv(lambda, y);
        x = floorf(x);
        z = _philox_uniform(thread_offset, seed, iter_count, &rng_reservoir, &rng_used);
        v = (_cr_pgammainc(lambda, x + 1.0f) - _cr_pgammainc(lambda, x)) * 1.3f;
        z = z*v;
        t = (float)_cr_exp(-lambda + x * logl - (float)_cr_lgamma_integer((int)(1.0f + x)));
        if ((z < t) && (v>=1e-20))
            break;
    }
    result.y = (unsigned int)x;

    while (true) {
        y = _philox_uniform(thread_offset, seed, iter_count, &rng_reservoir, &rng_used);
        x = _cr_pgammaincinv(lambda, y);
        x = floorf(x);
        z = _philox_uniform(thread_offset, seed, iter_count, &rng_reservoir, &rng_used);
        v = (_cr_pgammainc(lambda, x + 1.0f) - _cr_pgammainc(lambda, x)) * 1.3f;
        z = z*v;
        t = (float)_cr_exp(-lambda + x * logl - (float)_cr_lgamma_integer((int)(1.0f + x)));
        if ((z < t) && (v>=1e-20))
            break;
    }
    result.z = (unsigned int)x;

    while (true) {
        y = _philox_uniform(thread_offset, seed, iter_count, &rng_reservoir, &rng_used);
        x = _cr_pgammaincinv(lambda, y);
        x = floorf (x);
        z = _philox_uniform(thread_offset, seed, iter_count, &rng_reservoir, &rng_used);
        v = (_cr_pgammainc(lambda, x + 1.0f) - _cr_pgammainc(lambda, x)) * 1.3f;
        z = z*v;
        t = (float)_cr_exp(-lambda + x * logl - (float)_cr_lgamma_integer((int)(1.0f + x)));
        if ((z < t) && (v>=1e-20))
            break;
    }
    result.w = (unsigned int)x;

    return result;
}

QUALIFIERS
uint4
philox_poisson(
    uint32_t thread_offset,
    uint32_t seed,
    uint32_t *iter_count,
    double lambda
) {
    if (lambda < 64)
        return _philox_poisson_knuth4(thread_offset, seed, iter_count, (float)lambda);
    if (lambda > 4000) {
        double4 _res = philox_normal4_double(thread_offset, seed, iter_count);
        uint4 result;
        result.x = (unsigned int)((sqrt(lambda) * _res.x) + lambda + 0.5); //Round to nearest
        result.y = (unsigned int)((sqrt(lambda) * _res.y) + lambda + 0.5); //Round to nearest
        result.z = (unsigned int)((sqrt(lambda) * _res.z) + lambda + 0.5); //Round to nearest
        result.w = (unsigned int)((sqrt(lambda) * _res.w) + lambda + 0.5); //Round to nearest
    	return result;
    }
    return _philox_poisson_gammainc4(thread_offset, seed, iter_count, (float)lambda);
}
