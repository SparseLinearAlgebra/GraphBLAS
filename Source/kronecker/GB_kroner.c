//------------------------------------------------------------------------------
// GB_kroner: Kronecker product, C = kron (A,B)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = kron(A,B) where op determines the binary multiplier to use.  The type of
// C is the ztype of the operator.  C is hypersparse if either A or B are
// hypersparse, full if both A and B are full, or sparse otherwise.  C is never
// constructed as bitmap.

#define GB_FREE_WORKSPACE       \
{                               \
    GB_Matrix_free (&Awork) ;   \
    GB_Matrix_free (&Bwork) ;   \
}

#define GB_FREE_ALL             \
{                               \
    GB_FREE_WORKSPACE ;         \
    GB_phybix_free (C) ;        \
}

#define GBI(Ai,p,avlen) ((Ai == NULL) ? ((p) % (avlen)) : Ai [p])

#define GBB(Ab,p)       ((Ab == NULL) ? 1 : Ab [p])

#define GBP(Ap,k,avlen) ((Ap == NULL) ? ((k) * (avlen)) : Ap [k])

#define GBH(Ah,k)       ((Ah == NULL) ? (k) : Ah [k])

#include "mxm/GB_mxm.h"

static bool GB_lookup_xoffset (
    GrB_Index* p,
    GrB_Matrix A,
    GrB_Index row,
    GrB_Index col
)
{
    GrB_Index vector = A->is_csc ? col : row ;
    GrB_Index coord  = A->is_csc ? row : col ;

    if (A->p == NULL)
    {
        GrB_Index offset = vector * A->vlen + coord ;
        if (A->b == NULL || ((int8_t*)A->b)[offset])
        {
            *p = A->iso ? 0 : offset ;
            return true ;
        }
        return false ;
    }

    int64_t start, end ;
    bool res ;

    if (A->h == NULL)
    {
        start = A->p_is_32 ? ((uint32_t*)A->p)[vector] : ((uint64_t*)A->p)[vector] ;
        end = A->p_is_32 ? ((uint32_t*)A->p)[vector + 1] : ((uint64_t*)A->p)[vector + 1] ;
        end-- ;
        if (start > end) return false ;
        res = GB_binary_search(coord, A->i, A->i_is_32, &start, &end) ;
        if (res) { *p = A->iso ? 0 : start ; }
        return res ;
    }
    else
    {
        start = 0 ; end = A->plen - 1 ;
        res = GB_binary_search(vector, A->h, A->j_is_32, &start, &end) ;
        if (!res) return false ;
        int64_t k = start ;
        start = A->p_is_32 ? ((uint32_t*)A->p)[k] : ((uint64_t*)A->p)[k] ;
        end = A->p_is_32 ? ((uint32_t*)A->p)[k+1] : ((uint64_t*)A->p)[k+1] ;
        end-- ;
        if (start > end) return false ;
        res = GB_binary_search(coord, A->i, A->i_is_32, &start, &end) ;
        if (res) { *p = A->iso ? 0 : start ; }
        return res ;
    }
}


#include "kronecker/GB_kron.h"
#include "emult/GB_emult.h"
#include "slice/include/GB_search_for_vector.h"
#include "jitifyer/GB_stringify.h"

GrB_Info GB_kroner                  // C = kron (A,B)
(
    GrB_Matrix C,                   // output matrix
    const bool C_is_csc,            // desired format of C
    const GrB_BinaryOp op,          // multiply operator
    const bool flipij,              // if true, i and j are flipped: z=(x,y,j,i)
    const GrB_Matrix A_in,          // input matrix
    bool A_is_pattern,              // true if values of A are not used
    bool A_transpose,
    const GrB_Matrix B_in,          // input matrix
    bool B_is_pattern,              // true if values of B are not used
    bool B_transpose,
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (C != NULL && (C->header_size == 0 || GBNSTATIC)) ;

    struct GB_Matrix_opaque Awork_header, Bwork_header ;
    GrB_Matrix Awork = NULL, Bwork = NULL ;

    ASSERT_MATRIX_OK (A_in, "A_in for kron (A,B)", GB0) ;
    ASSERT_MATRIX_OK (B_in, "B_in for kron (A,B)", GB0) ;
    ASSERT_BINARYOP_OK (op, "op for kron (A,B)", GB0) ;

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (A_in) ;
    GB_MATRIX_WAIT (B_in) ;

    //--------------------------------------------------------------------------
    // can apply mask
    //--------------------------------------------------------------------------

    if (M != NULL && !Mask_comp)
    {
        GrB_Matrix A = A_in ;
        GrB_Matrix B = B_in ;
        GrB_Matrix MT = C ;

        GB_MATRIX_WAIT(M);

        int64_t bnrows = (B_transpose) ? GB_NCOLS (B) : GB_NROWS (B) ;
        int64_t bncols = (B_transpose) ? GB_NROWS (B) : GB_NCOLS (B) ;
        size_t allocated = 0 ;
        bool MT_hypersparse = (A->h != NULL) || (B->h != NULL);
        int64_t centries ;
        uint64_t nvecs ;
        centries = 0 ;
        nvecs = 0 ;

        uint32_t* MTp32 = NULL ; uint64_t* MTp64 = NULL ;
        MTp32 = M->p_is_32 ? GB_calloc_memory (M->vdim + 1, sizeof(uint32_t), &allocated) : NULL ;
        MTp64 = M->p_is_32 ? NULL : GB_calloc_memory (M->vdim + 1, sizeof(uint64_t), &allocated) ;
        if (MTp32 == NULL && MTp64 == NULL)
        {
            OUT_OF_MEM_p:
            GB_FREE_WORKSPACE ;
            return GrB_OUT_OF_MEMORY ;
        }

        GrB_Type MTtype = op->ztype ;
        const size_t MTsize = MTtype->size ;
        GB_void MTscalar [GB_VLA(MTsize)] ;
        bool MTiso =  GB_emult_iso (MTscalar, MTtype, A, B, op) ;

        GB_Mp_DECLARE(Mp, ) ;
        GB_Mp_PTR(Mp, M) ;

        GB_Mh_DECLARE(Mh, ) ;
        GB_Mh_PTR(Mh, M) ;

        GB_Mi_DECLARE(Mi, ) ;
        GB_Mi_PTR(Mi, M) ;

        GB_cast_function cast_A = NULL ;
        GB_cast_function cast_B = NULL ;
        
        cast_A = GB_cast_factory (op->xtype->code, A->type->code) ;
        cast_B = GB_cast_factory (op->ytype->code, B->type->code) ;

        double work = M->vdim ;
        int nthreads_max = GB_Context_nthreads_max ( ) ;
        double chunk = GB_Context_chunk ( ) ;
        int masked_nthreads = GB_nthreads (work, chunk, nthreads_max) ;

        int64_t vlen = M->vlen ;
        #pragma omp parallel num_threads(masked_nthreads)
        {
            GrB_Index offset ;

            #pragma omp for reduction(+:nvecs) schedule(static)
            for (GrB_Index k = 0 ; k < M->nvec ; k++)
            {
                GrB_Index j = Mh32 ? GBH (Mh32, k) : GBH (Mh64, k) ;

                int64_t pA_start = Mp32 ? GBP (Mp32, k, vlen) : GBP(Mp64, k, vlen) ;
                int64_t pA_end = Mp32 ? GBP (Mp32, k+1, vlen) : GBP(Mp64, k+1, vlen) ;
                bool nonempty = false ;
                for (GrB_Index p = pA_start ; p < pA_end ; p++)
                {
                    if (!GBB (M->b, p)) continue ;

                    int64_t i = Mi32 ? GBI (Mi32, p, vlen) : GBI (Mi64, p, vlen) ;
                    GrB_Index Mrow = M->is_csc ? i : j ; GrB_Index Mcol = M->is_csc ? j : i ;

                    // extract elements from A and B, increment MTp

                    if (Mask_struct || (M->iso ? ((int8_t*)M->x)[0] : ((int8_t*)M->x)[p]))
                    {
                        GrB_Index arow = A_transpose ? (Mcol / bncols) : (Mrow / bnrows) ;
                        GrB_Index acol = A_transpose ? (Mrow / bnrows) : (Mcol / bncols) ;

                        GrB_Index brow = B_transpose ? (Mcol % bncols) : (Mrow % bnrows) ;
                        GrB_Index bcol = B_transpose ? (Mrow % bnrows) : (Mcol % bncols) ;

                        bool code = GB_lookup_xoffset(&offset, A, arow, acol) ;
                        if (!code)
                        {
                            continue;
                        }

                        code = GB_lookup_xoffset(&offset, B, brow, bcol) ;
                        if (!code)
                        {
                            continue;
                        }

                        if (M->p_is_32)
                        {
                            (MTp32[j])++ ;
                        }
                        else
                        {
                            (MTp64[j])++ ;
                        }
                        nonempty = true ;
                    }
                }
                if (nonempty) nvecs++ ;
            }
        }

        // GB_cumsum for MT->p

        double work = M->vdim ;
        int nthreads_max = GB_Context_nthreads_max ( ) ;
        double chunk = GB_Context_chunk ( ) ;
        int cumsum_threads = GB_nthreads (work, chunk, nthreads_max) ;
        M->p_is_32 ? GB_cumsum(MTp32, M->p_is_32, M->vdim, NULL, cumsum_threads, Werk) :
        GB_cumsum(MTp64, M->p_is_32, M->vdim, NULL, cumsum_threads, Werk) ;

        centries = M->p_is_32 ? MTp32[M->vdim] : MTp64[M->vdim] ;

        uint32_t* MTi32 = NULL ; uint64_t* MTi64 = NULL;
        MTi32 = M->i_is_32 ? GB_malloc_memory (centries, sizeof(uint32_t), &allocated) : NULL ;
        MTi64 = M->i_is_32 ? NULL : GB_malloc_memory (centries, sizeof(uint64_t), &allocated) ;

        if (centries > 0 && MTi32 == NULL && MTi64 == NULL)
        {
            OUT_OF_MEM_i:
            if (M->p_is_32) { GB_free_memory (&MTp32, (M->vdim + 1) * sizeof(uint32_t)) ; }
            else { GB_free_memory (&MTp64, (M->vdim + 1) * sizeof(uint64_t)) ; }
            goto OUT_OF_MEM_p ;
        }

        void* MTx = NULL ;
        if (!MTiso)
        {
            MTx = GB_malloc_memory (centries, op->ztype->size, &allocated) ;
        }
        else
        {
            MTx = GB_malloc_memory (1, op->ztype->size, &allocated) ;
            if (MTx == NULL) goto OUT_OF_MEM_x ;
            memcpy (MTx, MTscalar, MTsize) ;
        }

        if (centries > 0 && MTx == NULL)
        {
            OUT_OF_MEM_x:
            if (M->i_is_32) { GB_free_memory (&MTi32, centries * sizeof(uint32_t)) ; }
            else { GB_free_memory (&MTi64, centries * sizeof (uint64_t)) ; }
            goto OUT_OF_MEM_i ;
        }

        #pragma omp parallel num_threads(masked_nthreads)
        {
            GrB_Index offset ;
            GB_void a_elem[op->xtype->size] ;
            GB_void b_elem[op->ytype->size] ;

            #pragma omp for schedule(static)
            for (GrB_Index k = 0 ; k < M->nvec ; k++)
            {
                GrB_Index j = Mh32 ? GBH (Mh32, k) : GBH (Mh64, k) ;

                int64_t pA_start = Mp32 ? GBP (Mp32, k, vlen) : GBP(Mp64, k, vlen) ;
                int64_t pA_end = Mp32 ? GBP (Mp32, k+1, vlen) : GBP(Mp64, k+1, vlen) ;
                GrB_Index pos = M->p_is_32 ? MTp32[j] : MTp64[j] ;
                for (GrB_Index p = pA_start ; p < pA_end ; p++)
                {
                    if (!GBB (M->b, p)) continue ;

                    int64_t i = Mi32 ? GBI (Mi32, p, vlen) : GBI (Mi64, p, vlen) ;
                    GrB_Index Mrow = M->is_csc ? i : j ; GrB_Index Mcol = M->is_csc ? j : i ;

                    // extract elements from A and B, 
                    // initialize offset in MTi and MTx,
                    // get result of op, place it in MTx

                    if (Mask_struct || (M->iso ? ((int8_t*)M->x)[0] : ((int8_t*)M->x)[p]))
                    {
                        GrB_Index arow = A_transpose ? (Mcol / bncols) : (Mrow / bnrows);
                        GrB_Index acol = A_transpose ? (Mrow / bnrows) : (Mcol / bncols);

                        GrB_Index brow = B_transpose ? (Mcol % bncols) : (Mrow % bnrows);
                        GrB_Index bcol = B_transpose ? (Mrow % bnrows) : (Mcol % bncols);

                        bool code = GB_lookup_xoffset (&offset, A, arow, acol) ;
                        if (!code)
                        {
                            continue;
                        }
                        if (!MTiso)
                        cast_A (a_elem, A->x + offset * A->type->size, A->type->size) ;

                        code = GB_lookup_xoffset (&offset, B, brow, bcol) ;
                        if (!code)
                        {
                            continue;
                        }
                        if (!MTiso)
                        cast_B (b_elem, B->x + offset * B->type->size, B->type->size) ;

                        if (!MTiso)
                        {
                            if (op->binop_function)
                            {
                                op->binop_function (MTx + op->ztype->size * pos, a_elem, b_elem) ;
                            }
                            else
                            {
                                GrB_Index ix, iy, jx, jy ;
                                ix = A_transpose ? acol : arow ;
                                iy = A_transpose ? arow : acol ;
                                jx = B_transpose ? bcol : brow ;
                                jy = B_transpose ? brow : bcol ;
                                op->idxbinop_function (MTx + op->ztype->size * pos, a_elem, ix, iy,
                                    b_elem, jx, jy, op->theta) ;
                            }
                        }

                        if (M->i_is_32) { MTi32[pos] = i ; } else { MTi64[pos] = i ; }
                        pos++ ;
                    }
                }
            }
        }

        #undef GBI
        #undef GBB
        #undef GBP
        #undef GBH

        // initialize other fields of MT properly

        GrB_Info MTalloc = GB_new_bix (&MT, op->ztype, vlen, M->vdim, GB_ph_null, M->is_csc, 
        GxB_SPARSE, false, M->hyper_switch, M->vdim, centries, false, MTiso, 
        M->p_is_32, M->j_is_32, M->i_is_32) ;
        if (MTalloc != GrB_SUCCESS)
        {
            if (MTiso) { GB_free_memory (&MTx, op->ztype->size) ; }
            else { GB_free_memory (&MTx, centries * op->ztype->size) ; }
            goto OUT_OF_MEM_x ;
        }

        GB_free_memory (&MT->i, MT->i_size) ;
        GB_free_memory (&MT->x, MT->x_size) ;

        MT->p = M->p_is_32 ? (void*)MTp32 : (void*)MTp64 ;
        MT->i = M->i_is_32 ? (void*)MTi32 : (void*)MTi64 ;
        MT->x = MTx ;

        MT->p_size = (M->p_is_32 ? sizeof(uint32_t) : sizeof(uint64_t)) * (M->vdim + 1) ;
        MT->i_size = ((M->i_is_32 ? sizeof(uint32_t) : sizeof(uint64_t)) * centries) ;
        MT->x_size = MT->iso ? op->ztype->size : op->ztype->size * centries ;
        MT->magic = GB_MAGIC ;
        MT->nvals = centries ;
        MT->nvec_nonempty = nvecs ;

        return GrB_SUCCESS ;
    }

    //--------------------------------------------------------------------------
    // bitmap case: create sparse copies of A and B if they are bitmap
    //--------------------------------------------------------------------------

    GrB_Matrix A = A_in ;
    if (GB_IS_BITMAP (A))
    { 
        GBURBLE ("A:") ;
        GB_CLEAR_MATRIX_HEADER (Awork, &Awork_header) ;
        GB_OK (GB_dup_worker (&Awork, A->iso, A, true, NULL)) ;
        ASSERT_MATRIX_OK (Awork, "dup Awork for kron (A,B)", GB0) ;
        GB_OK (GB_convert_bitmap_to_sparse (Awork, Werk)) ;
        ASSERT_MATRIX_OK (Awork, "to sparse, Awork for kron (A,B)", GB0) ;
        A = Awork ;
    }

    GrB_Matrix B = B_in ;
    if (GB_IS_BITMAP (B))
    { 
        GBURBLE ("B:") ;
        GB_CLEAR_MATRIX_HEADER (Bwork, &Bwork_header) ;
        GB_OK (GB_dup_worker (&Bwork, B->iso, B, true, NULL)) ;
        ASSERT_MATRIX_OK (Bwork, "dup Bwork for kron (A,B)", GB0) ;
        GB_OK (GB_convert_bitmap_to_sparse (Bwork, Werk)) ;
        ASSERT_MATRIX_OK (Bwork, "to sparse, Bwork for kron (A,B)", GB0) ;
        B = Bwork ;
    }

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;

    const int64_t avlen = A->vlen ;
    const int64_t avdim = A->vdim ;
    const int64_t anvec = A->nvec ;
    const int64_t anz = GB_nnz (A) ;

    GB_Bp_DECLARE (Bp, const) ; GB_Bp_PTR (Bp, B) ;
    GB_Bh_DECLARE (Bh, const) ; GB_Bh_PTR (Bh, B) ;

    const int64_t bvlen = B->vlen ;
    const int64_t bvdim = B->vdim ;
    const int64_t bnvec = B->nvec ;
    const int64_t bnz = GB_nnz (B) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    double work = ((double) anz) * ((double) bnz)
                + (((double) anvec) * ((double) bnvec)) ;

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (work, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // check if C is iso and compute its iso value if it is
    //--------------------------------------------------------------------------

    GrB_Type ctype = op->ztype ;
    const size_t csize = ctype->size ;
    GB_void cscalar [GB_VLA(csize)] ;
    bool C_iso = GB_emult_iso (cscalar, ctype, A, B, op) ;

    //--------------------------------------------------------------------------
    // allocate the output matrix C
    //--------------------------------------------------------------------------

    // C has the same type as z for the multiply operator, z=op(x,y)

    uint64_t cvlen, cvdim, cnzmax, cnvec ;
    bool ok = GB_int64_multiply (&cvlen, avlen, bvlen) ;
    ok = ok & GB_int64_multiply (&cvdim, avdim, bvdim) ;
    ok = ok & GB_int64_multiply (&cnzmax, anz, bnz) ;
    ok = ok & GB_int64_multiply (&cnvec, anvec, bnvec) ;
    ASSERT (ok) ;

    if (C_iso)
    { 
        // the values of A and B are no longer needed if C is iso
        GBURBLE ("(iso kron) ") ;
        A_is_pattern = true ;
        B_is_pattern = true ;
    }

    // C is hypersparse if either A or B are hypersparse.  It is never bitmap.
    bool C_is_hyper = (cvdim > 1) && (Ah != NULL || Bh != NULL) ;
    bool C_is_full = GB_as_if_full (A) && GB_as_if_full (B) ;
    int C_sparsity = C_is_full ? GxB_FULL :
        ((C_is_hyper) ? GxB_HYPERSPARSE : GxB_SPARSE) ;

    // determine the p_is_32, j_is_32, and i_is_32 settings for the new matrix

    bool Cp_is_32, Cj_is_32, Ci_is_32 ;
    GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
        C_sparsity, cnzmax, (int64_t) cvlen, (int64_t) cvdim, Werk) ;

    GB_OK (GB_new_bix (&C, // full, sparse, or hyper; existing header
        ctype, (int64_t) cvlen, (int64_t) cvdim, GB_ph_malloc, C_is_csc,
        C_sparsity, true, B->hyper_switch, cnvec, cnzmax, true, C_iso,
        Cp_is_32, Cj_is_32, Ci_is_32)) ;

    //--------------------------------------------------------------------------
    // compute the column counts of C: Cp and Ch if C is hypersparse
    //--------------------------------------------------------------------------

    GB_Cp_DECLARE (Cp, ) ; GB_Cp_PTR (Cp, C) ;
    GB_Ch_DECLARE (Ch, ) ; GB_Ch_PTR (Ch, C) ;
    #define GB_Cp_IS_32 Cp_is_32

    if (!C_is_full)
    { 
        // C is sparse or hypersparse
        int64_t kC ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (kC = 0 ; kC < cnvec ; kC++)
        {
            const int64_t kA = kC / bnvec ;
            const int64_t kB = kC % bnvec ;
            // get A(:,jA), the (kA)th vector of A
            const int64_t jA = GBh_A (Ah, kA) ;
            const int64_t aknz = (Ap == NULL) ? avlen :
                (GB_IGET (Ap, kA+1) - GB_IGET (Ap, kA)) ;
            // get B(:,jB), the (kB)th vector of B
            const int64_t jB = GBh_B (Bh, kB) ;
            const int64_t bknz = (Bp == NULL) ? bvlen :
                (GB_IGET (Bp, kB+1) - GB_IGET (Bp, kB)) ;
            // determine # entries in C(:,jC), the (kC)th vector of C
            // int64_t kC = kA * bnvec + kB ;
            // Cp [kC] = aknz * bknz ;
            GB_ISET (Cp, kC, aknz * bknz) ;
            if (C_is_hyper)
            { 
                // Ch [kC] = jA * bvdim + jB ;
                GB_ISET (Ch, kC, jA * bvdim + jB) ;
            }
        }

        int64_t nvec_nonempty ;
        GB_cumsum (Cp, Cp_is_32, cnvec, &nvec_nonempty, nthreads, Werk) ;
        GB_nvec_nonempty_set (C, nvec_nonempty) ;
        C->nvals = GB_IGET (Cp, cnvec) ;
        if (C_is_hyper) C->nvec = cnvec ;
    }

    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // C = kron (A,B) where C is iso and/or full full
    //--------------------------------------------------------------------------

    if (C_iso)
    { 
        // C->x [0] = cscalar = op (A,B)
        memcpy (C->x, cscalar, csize) ;
        if (C_is_full)
        { 
            // no more work to do if C is iso and full
            ASSERT_MATRIX_OK (C, "C=kron(A,B), iso full", GB0) ;
            GB_FREE_WORKSPACE ;
            return (GrB_SUCCESS) ;
        }
    }

    //--------------------------------------------------------------------------
    // quick return if C is empty
    //--------------------------------------------------------------------------

    int64_t cnz = GB_nnz (C) ;
    if (cnz == 0)
    { 
        GB_FREE_WORKSPACE ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // C = kron (A,B)
    //--------------------------------------------------------------------------

    // via the JIT kernel
    info = GB_kroner_jit (C, op, flipij, A, B, nthreads) ;

    if (info == GrB_NO_VALUE)
    { 
        // via the generic kernel
        #define GB_A_TYPE GB_void
        #define GB_B_TYPE GB_void
        #define GB_C_TYPE GB_void
        #define GB_A_ISO A_iso
        #define GB_B_ISO B_iso
        #define GB_C_ISO C_iso
        const bool A_iso = A->iso ;
        const bool B_iso = B->iso ;
        const int64_t asize = A->type->size ;
        const int64_t bsize = B->type->size ;

        GxB_binary_function fmult = op->binop_function ;
        GxB_index_binary_function fmult_idx = op->idxbinop_function ;
        const void *theta = op->theta ;
        GB_cast_function cast_A = NULL, cast_B = NULL ;
        if (!A_is_pattern)
        { 
            cast_A = GB_cast_factory (op->xtype->code, A->type->code) ;
        }
        if (!B_is_pattern)
        { 
            cast_B = GB_cast_factory (op->ytype->code, B->type->code) ;
        }

        #define GB_C_IS_FULL C_is_full

        #define GB_DECLAREA(a) GB_void a [GB_VLA(asize)]
        #define GB_DECLAREB(b) GB_void b [GB_VLA(bsize)]

        #define GB_GETA(a,Ax,p,iso)                         \
        {                                                   \
            if (!A_is_pattern)                              \
            {                                               \
                cast_A (a, Ax + (p)*asize, asize) ;         \
            }                                               \
        }

        #define GB_GETB(b,Bx,p,iso)                         \
        {                                                   \
            if (!B_is_pattern)                              \
            {                                               \
                cast_B (b, Bx + (p)*bsize, bsize) ;         \
            }                                               \
        }

        #define GB_KRONECKER_OP(Cx,pC,a,ix,jx,b,iy,jy)      \
        {                                                   \
            if (fmult != NULL)                              \
            {                                               \
                /* standard binary operator */              \
                fmult (Cx +(pC)*csize, a, b) ;              \
            }                                               \
            else                                            \
            {                                               \
                /* index binary operator */                 \
                if (flipij)                                 \
                {                                           \
                    fmult_idx (Cx +(pC)*csize,              \
                        a, jx, ix, b, jy, iy, theta) ;      \
                }                                           \
                else                                        \
                {                                           \
                    fmult_idx (Cx +(pC)*csize,              \
                        a, ix, jx, b, iy, jy, theta) ;      \
                }                                           \
            }                                               \
        }

        #define GB_GENERIC
        #include "ewise/include/GB_ewise_shared_definitions.h"
        #include "kronecker/template/GB_kroner_template.c"
        info = GrB_SUCCESS ;
    }

    //--------------------------------------------------------------------------
    // remove empty vectors from C, if hypersparse
    //--------------------------------------------------------------------------

    if (info == GrB_SUCCESS)
    { 
        GB_OK (GB_hyper_prune (C, Werk)) ;
        ASSERT_MATRIX_OK (C, "C=kron(A,B)", GB0) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    return (info) ;
}

