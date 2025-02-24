#= 
The Neighborhood Tensor Update (NTU) code is adapted from
YASTN (https://github.com/yastn/yastn).
Copyright 2024 The YASTN Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0
=#

"""
Neighborhood tensor update (NTU) algorithms to construct bond environment.
"""
abstract type NTUEnvAlgorithm end

"""
Construct the "NTU-NN" bond environment. 
```
            (-1 +0)══(-1 +1)
                ║        ║
    (+0 -1)═════X══   ═══Y═══(+0 +2)
                ║        ║
            (+1 +0)══(+1 +1)
```
"""
struct NTUEnvNN <: NTUEnvAlgorithm end
"""
Calculate the bond environment within "NTU-NN" approximation.
"""
function bondenv_ntu(
    row::Int, col::Int, X::T, Y::T, peps::InfiniteWeightPEPS, ::NTUEnvNN
) where {T<:Union{PEPSTensor,PEPSOrth}}
    neighbors = [
        (-1, 0, "tl"),
        (0, -1, "tbl"),
        (1, 0, "bl"),
        (1, 1, "rb"),
        (0, 2, "trb"),
        (-1, 1, "tr"),
    ]
    m = collect_neighbors(peps, row, col, neighbors)
    #= contraction indices

                (-1 +0) ══ Dt ══ (-1 +1)
                    ║               ║
            ........Dtl......       Dtr
                    ║       :       ║
        (+0 -1) ═══ X ══ Dl : Dr ══ Y ═══ (+0 +2)
                    ║       :       ║
                    Dbl     :.......Dbr........
                    ║               ║
                (+1 +0) ══ Db ══ (+1 +1)    
    =#
    # bottom-left half
    @autoopt @tensor benv_bl[Dbr1 Dbr0 Dl1 Dl0 Dtl1 Dtl0] :=
        cor_br(m[1, 1])[Dbr1 Dbr0 Db1 Db0] *
        cor_bl(m[1, 0])[Dbl1 Dbl0 Db1 Db0] *
        edge_l(X, hair_l(m[0, -1]))[Dtl1 Dtl0 Dl1 Dl0 Dbl1 Dbl0]
    benv_bl /= norm(benv_bl, Inf)
    # top-right half
    @autoopt @tensor benv_tr[Dtl1 Dtl0 Dr1 Dr0 Dbr1 Dbr0] :=
        cor_tl(m[-1, 0])[Dt1 Dt0 Dtl1 Dtl0] *
        cor_tr(m[-1, 1])[Dtr1 Dtr0 Dt1 Dt0] *
        edge_r(Y, hair_r(m[0, 2]))[Dtr1 Dtr0 Dbr1 Dbr0 Dr1 Dr0]
    benv_tr /= norm(benv_tr, Inf)
    @tensor benv[Dl1 Dr1; Dl0 Dr0] :=
        benv_bl[Dbr1 Dbr0 Dl1 Dl0 Dtl1 Dtl0] * benv_tr[Dtl1 Dtl0 Dr1 Dr0 Dbr1 Dbr0]
    @assert [isdual(space(benv, ax)) for ax in 1:4] == [0, 0, 1, 1]
    return benv / norm(benv, Inf)
end

"""
Construct the "NTU-NNN" bond environment. 
```
    (-1 -1)=(-1 +0)══(-1 +1)=(-1 +2)
        ║       ║        ║       ║
    (+0 -1)═════X══   ═══Y═══(+0 +2)
        ║       ║        ║       ║
    (+1 -1)=(+1 +0)══(+1 +1)=(+1 +2)
```
"""
struct NTUEnvNNN <: NTUEnvAlgorithm end
"""
Calculates the bond environment within "NTU-NNN" approximation.
"""
function bondenv_ntu(
    row::Int, col::Int, X::T, Y::T, peps::InfiniteWeightPEPS, ::NTUEnvNNN
) where {T<:Union{PEPSTensor,PEPSOrth}}
    neighbors = [
        (-1, -1, "tl"),
        (0, -1, "l"),
        (1, -1, "bl"),
        (1, 0, "b"),
        (1, 1, "b"),
        (1, 2, "rb"),
        (0, 2, "r"),
        (-1, 2, "tr"),
        (-1, 1, "t"),
        (-1, 0, "t"),
    ]
    m = collect_neighbors(peps, row, col, neighbors)
    #= left half
        (-1 -1)══════(-1 +0)═ -1/-2
            ║           ║
        (+0 -1)════════ X ═══ -3/-4
            ║           ║
        ....D1..........D2.........
            ║           ║
        (+1 -1)═ D3 ═(+1 +0)═ -5/-6
    =#
    vecl = enlarge_corner_tl(cor_tl(m[-1, -1]), edge_t(m[-1, 0]), edge_l(m[0, -1]), X)
    @tensor vecl[:] :=
        cor_bl(m[1, -1])[D11 D10 D31 D30] *
        edge_b(m[1, 0])[D21 D20 -5 -6 D31 D30] *
        vecl[D11 D10 D21 D20 -1 -2 -3 -4]
    vecl /= norm(vecl, Inf)
    #= right half
        -1/-2 ══ (-1 +1)═ D1 ═(-1 +2)
                    ║           ║
        ............D2..........D3...
                    ║           ║
        -3/-4 ═════ Y ═══════(+0 +2)
                    ║           ║     
        -5/-6 ══ (+1 +1)═════(+1 +2)
    =#
    vecr = enlarge_corner_br(cor_br(m[1, 2]), edge_b(m[1, 1]), edge_r(m[0, 2]), Y)
    @tensor vecr[:] :=
        edge_t(m[-1, 1])[D11 D10 D21 D20 -1 -2] *
        cor_tr(m[-1, 2])[D31 D30 D11 D10] *
        vecr[D21 D20 D31 D30 -3 -4 -5 -6]
    vecr /= norm(vecr, Inf)
    # combine left and right part
    @tensor benv[-1 -2; -3 -4] := vecl[1 2 -1 -3 3 4] * vecr[1 2 -2 -4 3 4]
    @assert [isdual(space(benv, ax)) for ax in 1:4] == [0, 0, 1, 1]
    return benv / norm(benv, Inf)
end

"""
Construct the "NTU-NNN+" bond environment. 
```
            (-2 -1) (-2 +0)  (-2 +1) (-2 +2)
                ║       ║        ║       ║
    (-1 -2)=(-1 -1)=(-1 +0)══(-1 +1)=(-1 +2)═(-1 +3)
                ║       ║        ║       ║
    (+0 -2)=(+0 -1)═════X══   ═══Y═══(+0 +2)═(+0 +3)
                ║       ║        ║       ║
    (+1 -2)=(+1 -1)=(+1 +0)══(+1 +1)═(+1 +2)═(+1 +3)
                ║       ║        ║       ║
            (+2 -1) (+2 +0)  (+2 +1) (+2 +2)
```
"""
struct NTUEnvNNNp <: NTUEnvAlgorithm end
"""
Calculates the bond environment within "NTU-NNN+" approximation.
"""
function bondenv_ntu(
    row::Int, col::Int, X::T, Y::T, peps::InfiniteWeightPEPS, ::NTUEnvNNNp
) where {T<:Union{PEPSTensor,PEPSOrth}}
    neighbors = [
        (-2, -1, "trl"),
        (-2, 0, "trl"),
        (-2, 1, "trl"),
        (-2, 2, "trl"),
        (-1, -2, "tbl"),
        (-1, -1, ""),
        (-1, 0, ""),
        (-1, 1, ""),
        (-1, 2, ""),
        (-1, 3, "trb"),
        (0, -2, "tbl"),
        (0, -1, ""),
        (0, 2, ""),
        (0, 3, "trb"),
        (1, -2, "tbl"),
        (1, -1, ""),
        (1, 0, ""),
        (1, 1, ""),
        (1, 2, ""),
        (1, 3, "trb"),
        (2, -1, "rbl"),
        (2, 0, "rbl"),
        (2, 1, "rbl"),
        (2, 2, "rbl"),
    ]
    m = collect_neighbors(peps, row, col, neighbors)
    #= left half
                (-2 -1)      (-2 +0)
                    ║           ║   
        (-1 -2)=(-1 -1)======(-1 +0)═ -1/-2
                    ║           ║
        (+0 -2)=(+0 -1)════════ X ═══ -3/-4
                    ║           ║
        ............D1..........D2.........
                    ║           ║
        (+1 -2)=(+1 -1)= D3 =(+1 +0)═ -5/-6
                    ║           ║
                (+2 -1)       (+2 +0)
    =#
    vecl = enlarge_corner_tl(
        cor_tl(m[-1, -1], hair_t(m[-2, -1]), hair_l(m[-1, -2])),
        edge_t(m[-1, 0], hair_t(m[-2, 0])),
        edge_l(m[0, -1], hair_l(m[0, -2])),
        X,
    )
    @tensor vecl[:] :=
        cor_bl(m[1, -1], hair_b(m[2, -1]), hair_l(m[1, -2]))[D11 D10 D31 D30] *
        edge_b(m[1, 0], hair_b(m[2, 0]))[D21 D20 -5 -6 D31 D30] *
        vecl[D11 D10 D21 D20 -1 -2 -3 -4]
    vecl /= norm(vecl, Inf)
    #= 
                (-2 +1)      (-2 +2)
                    ║           ║
        -1/-2 ══(-1 +1)═ D1 ═(-1 +2)═(-1 +3)
                    ║           ║
        ............D2..........D3..........
                    ║           ║
        -3/-4 ═════ Y ═══════(+0 +2)═(+0 +3)
                    ║           ║
        -5/-6 ══(+1 +1)══════(+1 +2)═(+1 +3)
                    ║           ║
                (+2 +1)      (+2 +2)
    =#
    vecr = enlarge_corner_br(
        cor_br(m[1, 2], hair_r(m[1, 3]), hair_b(m[2, 2])),
        edge_b(m[1, 1], hair_b(m[2, 1])),
        edge_r(m[0, 2], hair_r(m[0, 3])),
        Y,
    )
    @tensor vecr[:] :=
        edge_t(m[-1, 1], hair_t(m[-2, 1]))[D11 D10 D21 D20 -1 -2] *
        cor_tr(m[-1, 2], hair_t(m[-2, 2]), hair_r(m[-1, 3]))[D31 D30 D11 D10] *
        vecr[D21 D20 D31 D30 -3 -4 -5 -6]
    vecr /= norm(vecr, Inf)
    # combine left and right part
    @tensor benv[-1 -2; -3 -4] := vecl[1 2 -1 -3 3 4] * vecr[1 2 -2 -4 3 4]
    @assert [isdual(space(benv, ax)) for ax in 1:4] == [0, 0, 1, 1]
    return benv / norm(benv, Inf)
end
