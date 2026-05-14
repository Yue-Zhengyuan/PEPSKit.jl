"""
Extract bond tensor using `left_orth`
```
      ╲ ╱         ╲ ╱
    ---m---  =  ---X-←-  -←-q---
       ↓                    ↓
```
"""
function _bond_tensor_next(m::GenericMPSTensor{S, 4}) where {S}
    X, q = left_orth!(permute(m, ((1, 3, 4), (2, 5)); copy = true); positive = true)
    q = permute(q, ((1, 2), (3,)))
    return X, q
end

"""
Undo the decomposition in `_bond_tensor_next`.
"""
function _undo_bond_tensor_next(X::AbstractTensorMap{T, S, 3, 1}, q::MPSTensor) where {T, S}
    return @tensor m[-1 -2 -3 -4; -5] := X[-1 -3 -4; 1] * q[1 -2; -5]
end

"""
Extract bond tensor using `left_orth`
```
      ╲ ╱                  ╲ ╱
    ---m---  =  ---p-←-  -←-Y---
       ↓           ↓
```
"""
function _bond_tensor_prev(m::GenericMPSTensor{S, 4}) where {S}
    p, Y = right_orth!(permute(m, ((1, 2), (3, 4, 5)); copy = true); positive = true)
    return p, Y
end

function _undo_bond_tensor_prev(p::MPSTensor, Y::AbstractTensorMap{T, S, 1, 3}) where {T, S}
    return @tensor m[-1 -2 -3 -4; -5] := p[-1 -2; 1] * Y[1; -3 -4 -5]
end

"""
Calculate the reduced bond environment
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├--   --Y---b---┤
    |           ↓   |
    ├--   --Ȳ---b̄---┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
"""
function _benv_Yb(benv::BondEnv3site, Y::AbstractTensorMap{T, S, 1, 3}, b::MPSTensor) where {T, S}
    return @tensoropt benv_Yb[Da1 D1; Da0 D0] :=
        benv[Da1 De1 Ds1 Db1; Da0 De0 Ds0 Db0] *
        Y[D0; De0 Ds0 Dmb0] * b[Dmb0 db; Db0] *
        conj(Y[D1; De1 Ds1 Dmb1]) * conj(b[Dmb1 db; Db1])
end

"""
Calculate the reduced bond environment
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├---a---X--   --┤
    |   ↓           |
    ├---ā---X̄--   --┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
"""
function _benv_aX(benv::BondEnv3site, a::MPSTensor, X::AbstractTensorMap{T, S, 3, 1}) where {T, S}
    return @tensoropt benv_aX[D1 Db1; D0 Db0] :=
        benv[Da1 De1 Ds1 Db1; Da0 De0 Ds0 Db0] *
        a[Da0 da; Dam0] * X[Dam0 De0 Ds0; D0] *
        conj(a[Da1 da; Dam1]) * conj(X[Dam1 De1 Ds1; D1])
end

"""
Calculate the network
```
    ┌benv-┬---┬---------┐
    |      ╲ ╱          |
    ├---a₂==m₂======b₂--┤
    |   ↓   ↓       ↓   |
    ├--       --Ȳ---b̄---┤
    |          ╱ ╲      |
    └---------┴---┴-----┘
```
"""
function _benv_ket_Yb(
        benv_ket::AbstractTensorMap{T, S, 4, 3},
        Y::AbstractTensorMap{T, S, 1, 3}, b::MPSTensor
    ) where {T, S}
    return @tensoropt benv_ket_Yb[Da1 D1; da dm] :=
        benv_ket[Da1 De1 Ds1 Db1; da dm db] *
        conj(Y[D1; De1 Ds1 Dmb1]) * conj(b[Dmb1 db; Db1])
end

"""
Calculate the network
```
    ┌benv-----┬---┬-----┐
    |          ╲ ╱      |
    ├---a₂======m₂==b₂--┤
    |   ↓       ↓   ↓   |
    ├---ā---X--       --┤
    |      ╱ ╲          |
    └-----┴---┴---------┘
```
"""
function _benv_ket_aX(
        benv_ket::AbstractTensorMap{T, S, 4, 3},
        a::MPSTensor, X::AbstractTensorMap{T, S, 3, 1}
    ) where {T, S}
    return @tensoropt benv_ket_aX[D1 Db1; dm db] :=
        benv_ket[Da1 De1 Ds1 Db1; da dm db] *
        conj(a[Da1 da; Dam1]) * conj(X[Dam1 De1 Ds1; D1])
end
