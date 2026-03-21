//------------------------------------------------------------------------------
// GB_kroner_template: Kronecker product, C = kron (A,B)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = kron(A,B) where op determines the binary multiplier to use.  The type of
// C is the ztype of the operator.  C is hypersparse if either A or B are
// hypersparse, full if both A and B are full, or sparse otherwise.  C is never
// constructed as bitmap.  This template does not need access to C->h if C
// is hypersparse, so it works the same if C is sparse or hypersparse.

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_KERNEL
    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    const int64_t avlen = A->vlen ;

    GB_Bp_DECLARE (Bp, const) ; GB_Bp_PTR (Bp, B) ;
    GB_Bh_DECLARE (Bh, const) ; GB_Bh_PTR (Bh, B) ;
    const int64_t bvlen = B->vlen ;
    const int64_t bnvec = B->nvec ;

    #endif

    GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;
    GB_Bi_DECLARE (Bi, const) ; GB_Bi_PTR (Bi, B) ;

    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    const GB_B_TYPE *restrict Bx = (GB_A_TYPE *) B->x ;

    //--------------------------------------------------------------------------
    // count nonzero elements in result
    //--------------------------------------------------------------------------

    //----------------------------------------------------------------------
    // get the iso values of A and B
    //----------------------------------------------------------------------

    GB_DECLAREA (a) ;
    if (GB_A_ISO)
    { 
        GB_GETA (a, Ax, 0, true) ;
    }
    GB_DECLAREB (b) ;
    if (GB_B_ISO)
    { 
        GB_GETB (b, Bx, 0, true) ;
    }

    //----------------------------------------------------------------------
    // compute C(:,kC) for all vectors kC in this task
    //----------------------------------------------------------------------

    if (!GB_C_ISO && !OP_IS_POSITIONAL)
    {
        #pragma omp parallel for num_threads(nthreads) schedule(guided)
        for (int64_t kC = 0; kC < cnvec; kC++)
        {

            //------------------------------------------------------------------
            // get the vectors C(:,jC), A(:,jA), and B(:,jB)
            //------------------------------------------------------------------

            // C(:,jC) = kron (A(:,jA), B(:,jB), the (kC)th vector of C,
            // where jC = GBh_C (Ch, kC)
            int64_t kA = kC / bnvec ;
            int64_t kB = kC % bnvec ;

            // get A(:,jA), the (kA)th vector of A
            int64_t jA = GBh_A (Ah, kA) ;
            int64_t pA_start = GBp_A (Ap, kA, avlen) ;
            int64_t pA_end   = GBp_A (Ap, kA+1, avlen) ;

            // get B(:,jB), the (kB)th vector of B
            int64_t jB = GBh_B (Bh, kB) ;
            int64_t pB_start = GBp_B (Bp, kB, bvlen) ;
            int64_t pB_end   = GBp_B (Bp, kB+1, bvlen) ;
            int64_t bknz = pB_end - pB_start ;
            if (bknz == 0) continue ;

            //------------------------------------------------------------------
            // for all entries in A(:,jA), skipping entries for first vector
            //------------------------------------------------------------------

            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            {

                //--------------------------------------------------------------
                // a = A(iA,jA), typecasted to op->xtype
                //--------------------------------------------------------------

                int64_t iA = GBi_A (Ai, pA, avlen) ;
                int64_t iAblock = iA * bvlen ;
                if (!GB_A_ISO)
                { 
                    GB_GETA (a, Ax, pA, false) ;
                }

                //--------------------------------------------------------------
                // for all entries in B(:,jB), skipping entries for 1st vector
                //--------------------------------------------------------------

                for (int64_t pB = pB_start ; pB < pB_end ; pB++)
                { 

                    //----------------------------------------------------------
                    // b = B(iB,jB), typecasted to op->ytype
                    //----------------------------------------------------------

                    int64_t iB = GBi_B (Bi, pB, bvlen) ;
                    if (!GB_B_ISO)
                    { 
                        GB_GETB (b, Bx, pB, false) ;
                    }

                    //----------------------------------------------------------
                    // C(iC,jC) = A(iA,jA) * B(iB,jB)
                    //----------------------------------------------------------

                    if (!GB_C_ISO && !OP_IS_POSITIONAL)
                    { 
                        GB_C_TYPE cwork[GB_VLA(csize)] ;
                        GB_KRONECKER_COUNTER (cwork, a, b) ;
                    }
                }
            }
        }
        GB_cumsum (p, false, cnvec, NULL, nthreads, Werk) ;
        if (!(GB_C_IS_FULL = (GB_C_IS_FULL && cnz == cnzmax)))
        { 
            if (GB_C_IS_HYPER)
            { 
                h = GB_MALLOC_MEMORY (cnvec, sizeof(int64_t), &(h_size)) ;
                hp = GB_MALLOC_MEMORY (cnvec, sizeof(int64_t), &(hp_size)) ;
                ASSERT (h_size == GB_Global_memtable_size (h) && hp_size == GB_Global_memtable_size (hp)) ;
                GB_memset (h, 0, h_size, nthreads) ;
                GB_memset (hp, 0, hp_size, nthreads) ;
                for (kC = 0 ; kC < cnvec ; kC++)
                { 
                    if (p [kC + 1] > p [kC])
                    { 
                        int64_t kA = kC / bnvec ;
                        int64_t kB = kC % bnvec ;
                        const int64_t jA = GBh_A (Ah, kA) ;
                        const int64_t jB = GBh_B (Bh, kB) ;

                        h [nvec_nonempty++] = jA * bvdim + jB ;
                        hp [nvec_nonempty] = p [kC + 1] ;
                    }
                }
                cnz = hp[nvec_nonempty] ;
            }
            else
            {
                cnz = p[cnvec] ;
            }
        }
    }

    else if (!GB_C_IS_FULL)
    {
        h = GB_MALLOC_MEMORY (cnvec, sizeof(int64_t), &(h_size)) ;
        ASSERT (h_size == GB_Global_memtable_size (h)) ;
        #pragma omp parallel for num_threads(nthreads) schedule(guided)
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

            p [kC] = aknz * bknz ;

            if (C_is_hyper)
            { 
                h [kC] = jA * bvdim + jB ;
            }
        }

        GB_cumsum (p, false, cnvec, &(C->nvec_nonempty), nthreads, Werk) ;
        cnz = p[cnvec] ;
        if (GB_C_IS_HYPER) nvec_nonempty = cnvec ;
    }
}

