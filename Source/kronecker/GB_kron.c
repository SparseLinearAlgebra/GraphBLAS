//------------------------------------------------------------------------------
// GB_kron: C<M> = accum (C, kron(A,B))
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M> = accum (C, kron(A,B))

// The input matrices A and B are optionally transposed.

#define GB_FREE_WORKSPACE   \
{                           \
    GB_Matrix_free (&AT) ;  \
    GB_Matrix_free (&BT) ;  \
}

#define GB_FREE_ALL         \
{                           \
    GB_FREE_WORKSPACE ;     \
    GB_Matrix_free (&T) ;   \
}

#define GBI(Ai,p,avlen) ((Ai == NULL) ? ((p) % (avlen)) : Ai [p])

#define GBB(Ab,p)       ((Ab == NULL) ? 1 : Ab [p])

#define GBP(Ap,k,avlen) ((Ap == NULL) ? ((k) * (avlen)) : Ap [k])

#define GBH(Ah,k)       ((Ah == NULL) ? (k) : Ah [k])

#include "kronecker/GB_kron.h"
#include "mxm/GB_mxm.h"
#include "transpose/GB_transpose.h"
#include "mask/GB_accum_mask.h"

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

#include "emult/GB_emult.h"

GrB_Info GB_kron                    // C<M> = accum (C, kron(A,B))
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_replace,           // if true, clear C before writing to it
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op_in,       // defines '*' for kron(A,B)
    const GrB_Matrix A,             // input matrix
    bool A_transpose,               // if true, use A' instead of A
    const GrB_Matrix B,             // input matrix
    bool B_transpose,               // if true, use B' instead of B
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // C may be aliased with M, A, and/or B

    GrB_Info info ;
    struct GB_Matrix_opaque T_header, AT_header, BT_header ;
    GrB_Matrix T = NULL, AT = NULL, BT = NULL ;
    GrB_BinaryOp op = op_in ;

    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_FAULTY_OR_POSITIONAL (accum) ;

    ASSERT_MATRIX_OK (C, "C input for GB_kron", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for GB_kron", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for GB_kron", GB0) ;
    ASSERT_BINARYOP_OK (op, "op for GB_kron", GB0) ;
    ASSERT_MATRIX_OK (A, "A for GB_kron", GB0) ;
    ASSERT_MATRIX_OK (B, "B for GB_kron", GB0) ;

    // check domains and dimensions for C<M> = accum (C,T)
    GB_OK (GB_compatible (C->type, C, M, Mask_struct, accum, op->ztype,
        Werk)) ;

    // T=op(A,B) via op operator, so A and B must be compatible with z=op(a,b)
    GB_OK (GB_BinaryOp_compatible (op, NULL, A->type, B->type, GB_ignore_code,
        Werk)) ;

    // delete any lingering zombies and assemble any pending tuples in A and B,
    // so that cnz = nnz(A) * nnz(B) can be computed.  Updates of C and M are
    // done after this check.
    GB_MATRIX_WAIT (A) ;
    GB_MATRIX_WAIT (B) ;

    // check the dimensions of C
    int64_t anrows = (A_transpose) ? GB_NCOLS (A) : GB_NROWS (A) ;
    int64_t ancols = (A_transpose) ? GB_NROWS (A) : GB_NCOLS (A) ;
    int64_t bnrows = (B_transpose) ? GB_NCOLS (B) : GB_NROWS (B) ;
    int64_t bncols = (B_transpose) ? GB_NROWS (B) : GB_NCOLS (B) ;
    uint64_t cnrows, cncols, cnz = 0 ;
    bool ok = GB_int64_multiply (&cnrows, anrows,  bnrows) ;
    ok = ok && GB_int64_multiply (&cncols, ancols,  bncols) ;
    ok = ok && GB_int64_multiply (&cnz, GB_nnz (A), GB_nnz (B)) ;
    if (!ok || GB_NROWS (C) != cnrows || GB_NCOLS (C) != cncols)
    { 
        GB_ERROR (GrB_DIMENSION_MISMATCH, "%s:\n"
            "output is " GBd "-by-" GBd "; must be " GBu "-by-" GBu "\n"
            "first input is " GBd "-by-" GBd "%s with " GBd " entries\n"
            "second input is " GBd "-by-" GBd "%s with " GBd " entries",
            ok ? "Dimensions not compatible:" : "Problem too large:",
            GB_NROWS (C), GB_NCOLS (C), cnrows, cncols,
            anrows, ancols, A_transpose ? " (transposed)" : "", GB_nnz (A),
            bnrows, bncols, B_transpose ? " (transposed)" : "", GB_nnz (B)) ;
    }

    // quick return if an empty mask is complemented
    GB_RETURN_IF_QUICK_MASK (C, C_replace, M, Mask_comp, Mask_struct) ;

    // check if it's possible to apply mask immediately in kron
    // TODO: make MT of same CSR/CSC format as C
    // TODO: MT should have its own 32/64 bitness controls
    // TODO: clear MT header

    bool Mask_is_applicable = M != NULL && !Mask_comp ;
    if (Mask_is_applicable) {
        bool MT_hypersparse = (A->h != NULL) || (B->h != NULL) ;
        size_t allocated = 0 ;

        GrB_Matrix MT = NULL ; struct GB_Matrix_opaque MT_header ;
        GB_CLEAR_MATRIX_HEADER (MT, &MT_header) ;

        bool A_is_pattern, B_is_pattern ;
        GB_binop_pattern (&A_is_pattern, &B_is_pattern, false, op->opcode) ;

        GB_kroner (MT, C->is_csc, op, false, A, A_is_pattern, A_transpose, B, B_is_pattern, B_transpose,
        M, Mask_comp, Mask_struct, Werk) ;

        if (MT->is_csc != C->is_csc) {
            GrB_Info MTtranspose = GB_transpose_in_place (MT, true, Werk) ;
            if (MTtranspose != GrB_SUCCESS)
            {
                GB_FREE_WORKSPACE ;
                GB_Matrix_free (&MT) ;
                return MTtranspose ;
            }
        }

        if (MT_hypersparse)
        {
            uint32_t* MTh32 = NULL ; uint64_t* MTh64 =  NULL ;
            if (MT->j_is_32)
            {
                MTh32 = GB_malloc_memory (MT->vdim, sizeof(uint32_t), &allocated) ;
            }
            else
            {
                MTh64 = GB_malloc_memory (MT->vdim, sizeof(uint64_t), &allocated) ;
            }

            if (MTh32 == NULL && MTh64 == NULL)
            {
                GB_FREE_WORKSPACE ;
                GB_Matrix_free (&MT) ;
                return GrB_OUT_OF_MEMORY ;
            }

            #pragma omp parallel for
            for (GrB_Index i = 0; i < MT->vdim; i++)
            {
                if (MT->j_is_32) { MTh32[i] = i ; } else { MTh64[i] = i ; } 
            }

            MT->h = MTh32 ? (void*)MTh32 : (void*)MTh64 ;

            GrB_Info MThyperprune = GB_hyper_prune (MT, Werk) ;
            if (MThyperprune != GrB_SUCCESS)
            {
                GB_FREE_WORKSPACE ;
                GB_Matrix_free (&MT) ;
                return MThyperprune ;
            }
        }

        return (GB_accum_mask (C, NULL, NULL, accum, &MT, C_replace, Mask_comp, Mask_struct, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // transpose A and B if requested
    //--------------------------------------------------------------------------

    bool T_is_csc = C->is_csc ;
    if (T_is_csc != A->is_csc)
    { 
        // Negate A_transpose
        A_transpose = !A_transpose ;
    }
    if (T_is_csc != B->is_csc)
    { 
        // Negate B_transpose
        B_transpose = !B_transpose ;
    }

    // do not flipij the builtin positional ops (FIRSTI, and friends);
    // this is no longer needed with the new index binary ops.
    bool flipij = (!T_is_csc) ;

    bool A_is_pattern, B_is_pattern ;
    GB_binop_pattern (&A_is_pattern, &B_is_pattern, false, op->opcode) ;
    if (A_transpose)
    { 
        // AT = A' and typecast to op->xtype
        GBURBLE ("(A transpose) ") ;
        GB_CLEAR_MATRIX_HEADER (AT, &AT_header) ;
        GB_OK (GB_transpose_cast (AT, op->xtype, T_is_csc, A, A_is_pattern,
            Werk)) ;
        ASSERT_MATRIX_OK (AT, "AT kron", GB0) ;
    }

    if (B_transpose)
    { 
        // BT = B' and typecast to op->ytype
        GBURBLE ("(B transpose) ") ;
        GB_CLEAR_MATRIX_HEADER (BT, &BT_header) ;
        GB_OK (GB_transpose_cast (BT, op->ytype, T_is_csc, B, B_is_pattern,
            Werk)) ;
        ASSERT_MATRIX_OK (BT, "BT kron", GB0) ;
    }

    //--------------------------------------------------------------------------
    // T = kron(A,B)
    //--------------------------------------------------------------------------

    GB_CLEAR_MATRIX_HEADER (T, &T_header) ;
    GB_OK (GB_kroner (T, T_is_csc, op, flipij,
        A_transpose ? AT : A, A_is_pattern, A_transpose,
        B_transpose ? BT : B, B_is_pattern, B_transpose, M, Mask_comp, Mask_struct, Werk)) ;

    GB_FREE_WORKSPACE ;
    ASSERT_MATRIX_OK (T, "T = kron(A,B)", GB0) ;

    //--------------------------------------------------------------------------
    // C<M> = accum (C,T): accumulate the results into C via the mask
    //--------------------------------------------------------------------------

    return (GB_accum_mask (C, M, NULL, accum, &T, C_replace, Mask_comp,
        Mask_struct, Werk)) ;
}

