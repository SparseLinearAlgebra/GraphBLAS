function test227
%TEST227 test kron

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[binops, ~, ~, ~, ~, ~] = GB_spec_opsall ;
binops = binops.all ;

fprintf ('-------------- tests of GrB_kronecker:\n') ;

rng ('default') ;

dnn = struct ;
dtn = struct ( 'inp0', 'tran' ) ;
dnt = struct ( 'inp1', 'tran' ) ;
dtt = struct ( 'inp0', 'tran', 'inp1', 'tran' ) ;

dnn.mask = 'default' ;
dtn.mask = 'default' ;
dnt.mask = 'default' ;
dtt.mask = 'default' ;

dnn.outp = 'default' ;
dtn.outp = 'default' ;
dnt.outp = 'default' ;
dtt.outp = 'default' ;

types = { 'int32', 'int64', 'single', 'double' } ;

am = 5 ;
an = 3 ;
bm = 4 ;
bn = 2 ;

Ax_temp = 100 * sprandn (am, an, 0.5);
Bx_temp = 100 * sprandn (bm, bn, 0.5);

Ax = sparse(round(Ax_temp));
Bx = sparse(round(Bx_temp));

cm = am * bm ;
cn = an * bn ;
Cx = sparse (cm,cn) ;
Maskmat = sprandn (cm,cn,0.2) ~= 0 ;
ATmat = Ax' ;
BTmat = Bx' ;

for k2 = [4 7 45:52 ]
    for k1 = 1:4

        type = types {k1} ;
        binop = binops {k2}  ;
        op.opname = binop ;
        op.optype = type ;
        if (GB_spec_is_positional (op.opname))
            if (~(isequal (type, 'int32') || isequal (type, 'int64')))
                continue
            end
        end

        fprintf ('[ %s %s ] ', binop, type) ;

        for A_is_hyper = 0:1
            for A_is_csc   = 0:1
                for B_is_hyper = 0:1
                    for B_is_csc   = 0:1
                        for C_is_csc   = 0:1
                            fprintf ('.') ;

                            clear A
                            A.matrix = Ax ;
                            A.is_hyper = A_is_hyper ;
                            A.is_csc   = A_is_csc   ;

                            clear B
                            B.matrix = Bx ;
                            B.is_hyper = B_is_hyper ;
                            B.is_csc   = B_is_csc   ;

                            clear C
                            C.matrix = sparse (cm,cn) ;
                            C.is_csc   = C_is_csc   ;

                            clear AT
                            AT.matrix = ATmat ;
                            AT.is_hyper = A_is_hyper ;
                            AT.is_csc = A.is_csc ;

                            clear BT
                            BT.matrix = BTmat ;
                            BT.is_hyper = B_is_hyper ;
                            BT.is_csc = B.is_csc ;

                            %---------------------------------------
                            % kron(A,B)
                            %---------------------------------------

                            % C = kron(A,B)
                            C0 = GB_spec_kron (C, [ ], [ ], op, A, B, dnn) ;
                            C1 = GB_mex_kron  (C, [ ], [ ], op, A, B, dnn) ;
                            GB_spec_compare (C0, C1) ;

                            %---------------------------------------
                            % kron(A',B)
                            %---------------------------------------

                            % C = kron(A',B), no Mask
                            C0 = GB_spec_kron (C, [ ], [ ], op, AT, B, dtn) ;
                            C1 = GB_mex_kron  (C, [ ], [ ], op, AT, B, dtn) ;
                            GB_spec_compare (C0, C1) ;

                            %---------------------------------------
                            % kron(A,B')
                            %---------------------------------------

                            % no mask
                            C0 = GB_spec_kron (C, [ ], [ ], op, A, BT, dnt) ;
                            C1 = GB_mex_kron  (C, [ ], [ ], op, A, BT, dnt) ;
                            GB_spec_compare (C0, C1) ;

                            %---------------------------------------
                            % kron(A',B')
                            %---------------------------------------

                            % no Mask
                            C0 = GB_spec_kron (C, [ ], [ ], op, AT, BT, dtt) ;
                            C1 = GB_mex_kron  (C, [ ], [ ], op, AT, BT, dtt) ;
                            GB_spec_compare (C0, C1) ;

                            % tests with Mask
                            for Mask_is_hyper = 0:1
                            for Mask_is_csc = 0:1
                            fprintf('*')

                            A.is_csc = A_is_csc ;
                            B.is_csc = B_is_csc ;
                            A.is_hyper = A_is_hyper ;
                            B.is_hyper = B_is_hyper ;

                            clear M
                            M.matrix = Maskmat ;
                            M.is_hyper = Mask_is_hyper ;
                            M.is_csc = Mask_is_csc;
                            C.is_csc = C_is_csc ;

                            % kron(A, B) with Mask
                            C0 = GB_spec_kron (C, M, [ ], op, A, B, dnn) ;
                            fprintf('#') ;
                            C1 = GB_mex_kron (C, M, [ ], op, A, B, dnn) ;
                            GB_spec_compare(C0, C1) ;

                            % kron(A', B) with Mask
                            C0 = GB_spec_kron (C, M, [ ], op, AT, B, dtn) ;
                            C1 = GB_mex_kron  (C, M, [ ], op, AT, B, dtn) ;
                            GB_spec_compare (C0, C1) ;

                            % kron(A, B') with Mask
                            C0 = GB_spec_kron (C, M, [ ], op, A, BT, dnt) ;
                            C1 = GB_mex_kron  (C, M, [ ], op, A, BT, dnt) ;
                            GB_spec_compare (C0, C1) ;

                            % kron(A', B') with Mask
                            C0 = GB_spec_kron (C, M, [ ], op, AT, BT, dtt) ;
                            C1 = GB_mex_kron  (C, M, [ ], op, AT, BT, dtt) ;
                            GB_spec_compare (C0, C1) ;
                            end
                            end
                        end
                    end
                end
            end
        end
        fprintf ('\n') ;
    end
end

% test the empty case
clear op
op.opname = 'times' ;
op.optype = 'double' ;
clear A C
A.matrix = sparse (10,10) ;
A.pattern = zeros (10) ;
C.matrix = sparse (100,100) ;
C.pattern = zeros (100) ;
C0 = GB_spec_kron (C, [ ], [ ], op, A, A, dnn) ;
C1 = GB_mex_kron  (C, [ ], [ ], op, A, A, dnn) ;
GB_spec_compare (C0, C1) ;

fprintf ('\ntest227: all tests passed\n') ;

