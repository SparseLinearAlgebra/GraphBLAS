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
#include "kronecker/GB_kron.h"
#include "emult/GB_emult.h"
#include "slice/include/GB_search_for_vector.h"
#include "jitifyer/GB_stringify.h"

//------------------------------------------------------------------------------
// GB_lookup_xoffset: find the offset of (row,col) in a Ax if present
//------------------------------------------------------------------------------

static bool GB_lookup_xoffset
(
    GrB_Index *p,
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
        if (A->b == NULL || ((int8_t *) A->b) [offset])
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
        start = A->p_is_32 ? ((uint32_t *) A->p) [vector]
                           : ((uint64_t *) A->p) [vector] ;
        end   = A->p_is_32 ? ((uint32_t *) A->p) [vector + 1]
                           : ((uint64_t *) A->p) [vector + 1] ;
        end-- ;
        if (start > end) return false ;
        res = GB_binary_search (coord, A->i, A->i_is_32, &start, &end) ;
        if (res) { *p = A->iso ? 0 : start ; }
        return res ;
    }
    else
    {
        start = 0 ; end = A->plen - 1 ;
        res = GB_binary_search (vector, A->h, A->j_is_32, &start, &end) ;
        if (!res) return false ;
        int64_t k = start ;
        start = A->p_is_32 ? ((uint32_t *) A->p) [k]
                           : ((uint64_t *) A->p) [k] ;
        end   = A->p_is_32 ? ((uint32_t *) A->p) [k + 1]
                           : ((uint64_t *) A->p) [k + 1] ;
        end-- ;
        if (start > end) return false ;
        res = GB_binary_search (coord, A->i, A->i_is_32, &start, &end) ;
        if (res) { *p = A->iso ? 0 : start ; }
        return res ;
    }
}

//------------------------------------------------------------------------------
// GB_kroner_generic: generic kernel for masked and default paths
//------------------------------------------------------------------------------

static GrB_Info GB_kroner_generic
(
    GrB_Matrix C,
    const GrB_BinaryOp op,
    const bool flipij,
    const GrB_Matrix A,
    const GrB_Matrix B,
    bool A_is_pattern,
    bool B_is_pattern,
    bool A_transpose,
    bool B_transpose,
    const GrB_Matrix Mask,
    const bool Mask_comp,
    const bool Mask_struct,
    const bool C_is_full,
    const int nthreads
)
{
    GrB_Type ctype = op->ztype ;
    const size_t csize = ctype->size ;

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    const int64_t avlen = A->vlen ;

    GB_Bp_DECLARE (Bp, const) ; GB_Bp_PTR (Bp, B) ;
    GB_Bh_DECLARE (Bh, const) ; GB_Bh_PTR (Bh, B) ;
    const int64_t bvlen = B->vlen ;
    const int64_t bnvec = B->nvec ;

    GB_Cp_DECLARE (Cp, ) ; GB_Cp_PTR (Cp, C) ;
    const int64_t cnvec = C->nvec ;
    const int64_t cvlen = C->vlen ;
    const int64_t cnz   = C->nvals ;

    #define GB_NO_MASK      (Mask == NULL)
    #define GB_MASK_STRUCT  Mask_struct
    #define GB_MASK_COMP    Mask_comp
    #define GB_Cp_IS_32     C->p_is_32
    #define GB_A_TYPE       GB_void
    #define GB_B_TYPE       GB_void
    #define GB_C_TYPE       GB_void
    #define GB_A_ISO        A_iso
    #define GB_B_ISO        B_iso
    #define GB_C_ISO        C_iso
    #define GB_C_IS_FULL    C_is_full

    const bool A_iso    = A->iso ;
    const bool B_iso    = B->iso ;
    const bool C_iso    = C->iso ;
    const int64_t asize = A->type->size ;
    const int64_t bsize = B->type->size ;

    GxB_binary_function       fmult     = op->binop_function ;
    GxB_index_binary_function fmult_idx = op->idxbinop_function ;
    const void *theta = op->theta ;

    GB_cast_function cast_A = NULL, cast_B = NULL ;
    if (!A_is_pattern)
        cast_A = GB_cast_factory (op->xtype->code, A->type->code) ;
    if (!B_is_pattern)
        cast_B = GB_cast_factory (op->ytype->code, B->type->code) ;

    #define GB_DECLAREA(a) GB_void a [GB_VLA(asize)]
    #define GB_DECLAREB(b) GB_void b [GB_VLA(bsize)]

    #define GB_GETA(a,Ax,p,iso)                         \
    {                                                   \
        if (!A_is_pattern)                              \
            cast_A (a, Ax + (p)*asize, asize) ;         \
    }

    #define GB_GETB(b,Bx,p,iso)                         \
    {                                                   \
        if (!B_is_pattern)                              \
            cast_B (b, Bx + (p)*bsize, bsize) ;         \
    }

    #define GB_KRONECKER_OP(Cx,pC,a,ix,jx,b,iy,jy)      \
    {                                                   \
        if (fmult != NULL)                              \
        {                                               \
            fmult (Cx + (pC)*csize, a, b) ;             \
        }                                               \
        else                                            \
        {                                               \
            if (flipij)                                 \
                fmult_idx (Cx + (pC)*csize,             \
                    a, jx, ix, b, jy, iy, theta) ;      \
            else                                        \
                fmult_idx (Cx + (pC)*csize,             \
                    a, ix, jx, b, iy, jy, theta) ;      \
        }                                               \
    }

    #define GB_GENERIC
    #include "ewise/include/GB_ewise_shared_definitions.h"
    #include "kronecker/template/GB_kroner_template.c"

    return GrB_SUCCESS ;
}

//------------------------------------------------------------------------------
// GB_kroner: Kronecker product, C = kron (A,B)
//------------------------------------------------------------------------------

GrB_Info GB_kroner
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
    const GrB_Matrix Mask,
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
    // common definitions for masked and default
    //--------------------------------------------------------------------------

    GrB_Type ctype = op->ztype ;
    const size_t csize = ctype->size ;
    GB_void cscalar [GB_VLA(csize)] ;
    bool C_iso = GB_emult_iso (cscalar, ctype, A_in, B_in, op) ;

    GrB_Matrix A = A_in ;
    GrB_Matrix B = B_in ;

    //--------------------------------------------------------------------------
    // apply mask immediately if possible
    //--------------------------------------------------------------------------

    if (Mask != NULL && !Mask_comp)
    {
        GB_MATRIX_WAIT (Mask) ;

        int64_t bnrows = B_transpose ? GB_NCOLS (B) : GB_NROWS (B) ;
        int64_t bncols = B_transpose ? GB_NROWS (B) : GB_NCOLS (B) ;
        size_t allocated = 0 ;
        int64_t centries = 0 ;
        uint64_t nvecs = 0 ;

        //----------------------------------------------------------------------
        // allocate Cp
        //----------------------------------------------------------------------

        int64_t mnzmax = GB_nnz_max (Mask) ;
        bool Cp_is_32, Cj_is_32, Ci_is_32;
        GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
            GxB_SPARSE, mnzmax, (int64_t) Mask->vlen, (int64_t) Mask->vdim, Werk) ;

        uint32_t *Cp32 = NULL ; uint64_t *Cp64 = NULL ;
        if (Cp_is_32)
            Cp32 = GB_calloc_memory (Mask->vdim + 1, sizeof (uint32_t),
                &allocated) ;
        else
            Cp64 = GB_calloc_memory (Mask->vdim + 1, sizeof (uint64_t),
                &allocated) ;

        if (Cp32 == NULL && Cp64 == NULL)
        {
            GB_FREE_WORKSPACE ;
            return GrB_OUT_OF_MEMORY ;
        }

        GB_Mp_DECLARE (Mp, ) ; GB_Mp_PTR (Mp, Mask) ;
        GB_Mh_DECLARE (Mh, ) ; GB_Mh_PTR (Mh, Mask) ;
        GB_Mi_DECLARE (Mi, ) ; GB_Mi_PTR (Mi, Mask) ;

        GB_cast_function cast_A = GB_cast_factory (op->xtype->code,
            A->type->code) ;
        GB_cast_function cast_B = GB_cast_factory (op->ytype->code,
            B->type->code) ;

        double work = Mask->vdim ;
        int nthreads_max = GB_Context_nthreads_max ( ) ;
        double chunk = GB_Context_chunk ( ) ;
        int nthreads = GB_nthreads (work, chunk, nthreads_max) ;

        int64_t vlen = Mask->vlen ;

        //----------------------------------------------------------------------
        // count entries per vector
        //----------------------------------------------------------------------

        #pragma omp parallel num_threads(nthreads)
        {
            GrB_Index offset ;

            #pragma omp for reduction(+:nvecs) schedule(static)
            for (GrB_Index k = 0 ; k < Mask->nvec ; k++)
            {
                GrB_Index j = Mh32 ? GBH (Mh32, k) : GBH (Mh64, k) ;

                int64_t pM_start = Mp32 ? GBP (Mp32, k,   vlen)
                                        : GBP (Mp64, k,   vlen) ;
                int64_t pM_end   = Mp32 ? GBP (Mp32, k+1, vlen)
                                        : GBP (Mp64, k+1, vlen) ;
                bool nonempty = false ;

                for (GrB_Index p = pM_start ; p < pM_end ; p++)
                {
                    if (!GBB (Mask->b, p)) continue ;

                    int64_t i = Mi32 ? GBI (Mi32, p, vlen)
                                     : GBI (Mi64, p, vlen) ;
                    GrB_Index Mrow = Mask->is_csc ? i : j ;
                    GrB_Index Mcol = Mask->is_csc ? j : i ;

                    if (Mask_struct || (Mask->iso ? ((bool *) Mask->x) [0]
                                                  : ((bool *) Mask->x) [p]))
                    {
                        GrB_Index arow = A_transpose ? (Mcol / bncols)
                                                     : (Mrow / bnrows) ;
                        GrB_Index acol = A_transpose ? (Mrow / bnrows)
                                                     : (Mcol / bncols) ;
                        GrB_Index brow = B_transpose ? (Mcol % bncols)
                                                     : (Mrow % bnrows) ;
                        GrB_Index bcol = B_transpose ? (Mrow % bnrows)
                                                     : (Mcol % bncols) ;

                        if (!GB_lookup_xoffset (&offset, A, arow, acol))
                            continue ;
                        if (!GB_lookup_xoffset (&offset, B, brow, bcol))
                            continue ;

                        if (Cp_is_32)
                        {
                            (Cp32 [j])++ ;
                        }
                        else
                        {
                            (Cp64 [j])++ ;
                        }
                        nonempty = true ;
                    }
                }
                if (nonempty) nvecs++ ;
            }
        }

        //----------------------------------------------------------------------
        // prefix sum to get centries
        //----------------------------------------------------------------------

        if (Cp_is_32)
            GB_cumsum (Cp32, Cp_is_32, Mask->vdim, NULL, nthreads, Werk) ;
        else
            GB_cumsum (Cp64, Cp_is_32, Mask->vdim, NULL, nthreads, Werk) ;

        centries = Cp_is_32 ? (int64_t) Cp32 [Mask->vdim]
                            : (int64_t) Cp64 [Mask->vdim] ;

        //----------------------------------------------------------------------
        // allocate Ci
        //----------------------------------------------------------------------

        uint32_t *Ci32 = NULL ; uint64_t *Ci64 = NULL ;
        if (Ci_is_32)
            Ci32 = GB_malloc_memory (centries, sizeof (uint32_t), &allocated) ;
        else
            Ci64 = GB_malloc_memory (centries, sizeof (uint64_t), &allocated) ;

        if (centries > 0 && Ci32 == NULL && Ci64 == NULL)
        {
            if (Cp_is_32)
            {
                GB_free_memory (&Cp32, (Mask->vdim + 1) * sizeof (uint32_t)) ;
            }
            else
            {
                GB_free_memory (&Cp64, (Mask->vdim + 1) * sizeof (uint64_t)) ;
            }
            GB_FREE_WORKSPACE ;
            return GrB_OUT_OF_MEMORY ;
        }

        //----------------------------------------------------------------------
        // allocate Cx
        //----------------------------------------------------------------------

        void *Cx = NULL ;
        if (C_iso)
        {
            Cx = GB_malloc_memory (1, op->ztype->size, &allocated) ;
            if (Cx == NULL)
            {
                if (Ci_is_32)
                {
                    GB_free_memory (&Ci32, centries * sizeof (uint32_t)) ;
                }
                else
                {
                    GB_free_memory (&Ci64, centries * sizeof (uint64_t)) ;
                }
                if (Cp_is_32)
                {
                    GB_free_memory (&Cp32, (Mask->vdim + 1) * sizeof (uint32_t)) ;
                }
                else
                {
                    GB_free_memory (&Cp64, (Mask->vdim + 1) * sizeof (uint64_t)) ;
                }
                GB_FREE_WORKSPACE ;
                return GrB_OUT_OF_MEMORY ;
            }
            memcpy (Cx, cscalar, csize) ;
        }
        else
        {
            Cx = GB_malloc_memory (centries, op->ztype->size, &allocated) ;
            if (centries > 0 && Cx == NULL)
            {
                if (Ci_is_32)
                {
                    GB_free_memory (&Ci32, centries * sizeof (uint32_t)) ;
                }
                else
                {
                    GB_free_memory (&Ci64, centries * sizeof (uint64_t)) ;
                }
                if (Cp_is_32)
                {
                    GB_free_memory (&Cp32, (Mask->vdim + 1) * sizeof (uint32_t)) ;
                }
                else
                {
                    GB_free_memory (&Cp64, (Mask->vdim + 1) * sizeof (uint64_t)) ;
                }
                GB_FREE_WORKSPACE ;
                return GrB_OUT_OF_MEMORY ;
            }
        }

        //----------------------------------------------------------------------
        // allocate and initialize C matrix header
        //----------------------------------------------------------------------

        GrB_Info Calloc = GB_new_bix (&C, op->ztype, vlen, Mask->vdim,
            GB_ph_null, Mask->is_csc, GxB_SPARSE, false, Mask->hyper_switch,
            Mask->vdim, centries, false, C_iso,
            Cp_is_32, Cj_is_32, Ci_is_32) ;
        if (Calloc != GrB_SUCCESS)
        {
            if (C_iso)
            {
                GB_free_memory (&Cx, op->ztype->size) ;
            }
            else
            {
                GB_free_memory (&Cx, centries * op->ztype->size) ;
            }
            if (Ci_is_32)
            {
                GB_free_memory (&Ci32, centries * sizeof (uint32_t)) ;
            }
            else
            {
                GB_free_memory (&Ci64, centries * sizeof (uint64_t)) ;
            }
            if (Cp_is_32)
            {
                GB_free_memory (&Cp32, (Mask->vdim + 1) * sizeof (uint32_t)) ;
            }
            else
            {
                GB_free_memory (&Cp64, (Mask->vdim + 1) * sizeof (uint64_t)) ;
            }
            GB_FREE_WORKSPACE ;
            return Calloc ;
        }

        GB_free_memory (&C->i, C->i_size) ;
        GB_free_memory (&C->x, C->x_size) ;

        C->p = Cp_is_32 ? (void *) Cp32 : (void *) Cp64 ;
        C->i = Ci_is_32 ? (void *) Ci32 : (void *) Ci64 ;
        C->x = Cx ;
        C->p_size = (Cp_is_32 ? sizeof (uint32_t)
                              : sizeof (uint64_t)) * (Mask->vdim + 1) ;
        C->i_size = (Ci_is_32 ? sizeof (uint32_t)
                              : sizeof (uint64_t)) * centries ;
        C->x_size = C->iso ? op->ztype->size
                           : op->ztype->size * centries ;
        C->magic = GB_MAGIC ;
        C->nvals = centries ;
        C->nvec_nonempty = (int64_t) nvecs ;

        //----------------------------------------------------------------------
        // evaluate
        //----------------------------------------------------------------------

        info = GB_kroner_jit (C, op, false, A, A_transpose, B, B_transpose, Mask, Mask_struct, Mask_comp, nthreads) ;

        if (info != GrB_SUCCESS)
        {
            info = GB_kroner_generic (C, op, false, A, B,
                A_is_pattern, B_is_pattern, A_transpose, B_transpose,
                Mask, Mask_comp, Mask_struct, false, nthreads) ;
        }

        GB_FREE_WORKSPACE ;
        return info ;
    }

    //--------------------------------------------------------------------------
    // default case (no mask, or complemented mask)
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // bitmap case: create sparse copies of A and B if they are bitmap
    //--------------------------------------------------------------------------

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
    const int64_t anz   = GB_nnz (A) ;

    GB_Bp_DECLARE (Bp, const) ; GB_Bp_PTR (Bp, B) ;
    GB_Bh_DECLARE (Bh, const) ; GB_Bh_PTR (Bh, B) ;

    const int64_t bvlen = B->vlen ;
    const int64_t bvdim = B->vdim ;
    const int64_t bnvec = B->nvec ;
    const int64_t bnz   = GB_nnz (B) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    double work = ((double) anz) * ((double) bnz)
                + (((double) anvec) * ((double) bnvec)) ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (work, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // allocate the output matrix C
    //--------------------------------------------------------------------------

    uint64_t cvlen, cvdim, cnzmax, cnvec ;
    bool ok = GB_int64_multiply (&cvlen,  avlen, bvlen) ;
    ok = ok & GB_int64_multiply (&cvdim,  avdim, bvdim) ;
    ok = ok & GB_int64_multiply (&cnzmax, anz,   bnz) ;
    ok = ok & GB_int64_multiply (&cnvec,  anvec, bnvec) ;
    ASSERT (ok) ;

    if (C_iso)
    {
        GBURBLE ("(iso kron) ") ;
        A_is_pattern = true ;
        B_is_pattern = true ;
    }

    bool C_is_hyper = (cvdim > 1) && (Ah != NULL || Bh != NULL) ;
    bool C_is_full  = GB_as_if_full (A) && GB_as_if_full (B) ;
    int  C_sparsity = C_is_full  ? GxB_FULL :
                      C_is_hyper ? GxB_HYPERSPARSE : GxB_SPARSE ;

    bool Cp_is_32, Cj_is_32, Ci_is_32 ;
    GB_determine_pji_is_32 (&Cp_is_32, &Cj_is_32, &Ci_is_32,
        C_sparsity, cnzmax, (int64_t) cvlen, (int64_t) cvdim, Werk) ;

    GB_OK (GB_new_bix (&C, ctype, (int64_t) cvlen, (int64_t) cvdim,
        GB_ph_malloc, C_is_csc, C_sparsity, true, B->hyper_switch,
        cnvec, cnzmax, true, C_iso, Cp_is_32, Cj_is_32, Ci_is_32)) ;

    //--------------------------------------------------------------------------
    // compute the column counts of C: Cp and Ch if C is hypersparse
    //--------------------------------------------------------------------------

    GB_Cp_DECLARE (Cp, ) ; GB_Cp_PTR (Cp, C) ;
    GB_Ch_DECLARE (Ch, ) ; GB_Ch_PTR (Ch, C) ;
    //#define GB_Cp_IS_32 Cp_is_32

    if (!C_is_full)
    {
        int64_t kC ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (kC = 0 ; kC < (int64_t) cnvec ; kC++)
        {
            const int64_t kA = kC / bnvec ;
            const int64_t kB = kC % bnvec ;
            const int64_t jA = GBh_A (Ah, kA) ;
            const int64_t aknz = (Ap == NULL) ? avlen :
                (GB_IGET (Ap, kA+1) - GB_IGET (Ap, kA)) ;
            const int64_t jB = GBh_B (Bh, kB) ;
            const int64_t bknz = (Bp == NULL) ? bvlen :
                (GB_IGET (Bp, kB+1) - GB_IGET (Bp, kB)) ;
            GB_ISET (Cp, kC, aknz * bknz) ;
            if (C_is_hyper)
                GB_ISET (Ch, kC, jA * bvdim + jB) ;
        }

        int64_t nvec_nonempty ;
        GB_cumsum (Cp, Cp_is_32, cnvec, &nvec_nonempty, nthreads, Werk) ;
        GB_nvec_nonempty_set (C, nvec_nonempty) ;
        C->nvals = GB_IGET (Cp, cnvec) ;
        if (C_is_hyper) C->nvec = cnvec ;
    }

    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // C = kron (A,B) where C is iso and/or full
    //--------------------------------------------------------------------------

    if (C_iso)
    {
        memcpy (C->x, cscalar, csize) ;
        if (C_is_full)
        {
            ASSERT_MATRIX_OK (C, "C=kron(A,B), iso full", GB0) ;
            GB_FREE_WORKSPACE ;
            return GrB_SUCCESS ;
        }
    }

    //--------------------------------------------------------------------------
    // quick return if C is empty
    //--------------------------------------------------------------------------

    int64_t cnz = GB_nnz (C) ;
    if (cnz == 0)
    {
        GB_FREE_WORKSPACE ;
        return GrB_SUCCESS ;
    }

    //--------------------------------------------------------------------------
    // evaluate: JIT or generic
    //--------------------------------------------------------------------------

    info = GB_kroner_jit (C, op, flipij, A, A_transpose, B, B_transpose, Mask, Mask_struct, Mask_comp, nthreads) ;

    if (info != GrB_SUCCESS)
    {
        info = GB_kroner_generic (C, op, flipij, A, B,
            A_is_pattern, B_is_pattern, A_transpose, B_transpose,
            Mask, Mask_comp, Mask_struct, C_is_full, nthreads) ;
    }

    //--------------------------------------------------------------------------
    // remove empty vectors from C, if hypersparse
    //--------------------------------------------------------------------------

    if (info == GrB_SUCCESS)
    {
        GB_OK (GB_hyper_prune (C, Werk)) ;
        ASSERT_MATRIX_OK (C, "C=kron(A,B)", GB0) ;
    }

    GB_FREE_WORKSPACE ;
    return info ;
}
