function test226
%TEST226 test kron with iso matrices

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

A.matrix = sprand (5, 10, 0.4) ;
B.matrix = ones (3, 2) ;
B.iso = true ;
M.matrix = sprandn (15, 20,0.2) ~= 0 ;
MT.matrix = sprandn (9, 4, 20,0.2) ~= 0 ;

mult.opname = 'times' ;
mult.optype = 'double' ;

Cin = sparse (15, 20) ;
C1 = GB_mex_kron  (Cin, [ ], [ ], mult, A, B, [ ]) ;
C2 = GB_spec_kron (Cin, [ ], [ ], mult, A, B, [ ]) ;
GB_spec_compare (C1, C2) ;

C1 = GB_mex_kron (Cin, M, [ ], mult, A, B, [ ]) ;
C2 = GB_spec_kron (Cin, M, [ ], mult, A, B, [ ]) ;
GB_spec_compare (C1, C2) ;

C1 = GB_mex_kron  (Cin, [ ], [ ], mult, B, A, [ ]) ;
C2 = GB_spec_kron (Cin, [ ], [ ], mult, B, A, [ ]) ;
GB_spec_compare (C1, C2) ;

C1 = GB_mex_kron (Cin, M, [ ], mult, B, A, [ ]) ;
C2 = GB_spec_kron (Cin, M, [ ], mult, B, A, [ ]) ;
GB_spec_compare (C1, C2) ;

Cin = sparse (9, 4) ;
C1 = GB_mex_kron  (Cin, [ ], [ ], mult, B, B, [ ]) ;
C2 = GB_spec_kron (Cin, [ ], [ ], mult, B, B, [ ]) ;
GB_spec_compare (C1, C2) ;

C1 = GB_mex_kron (Cin, MT, [ ], mult, B, B, [ ]) ;
C2 = GB_spec_kron (Cin, MT, [ ], mult, B, B, [ ]) ;
GB_spec_compare (C1, C2) ;

fprintf ('\ntest226: all tests passed\n') ;

