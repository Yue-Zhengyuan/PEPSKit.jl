_tensor_halfR(Z, Ms, x::Int) = _tensor_halfR(Z, Ms, Val(x))
_tensor_S(hN, hR, x::Int) = _tensor_S(hN, hR, Val(x))

"""
$(SIGNATURES)

Construct the tensor
```
    ┌-------┐
    | ┌---┬-Z-┬---┐
    | |    ╲ ╱    |
    | └   --m---b-┘
    ↓       ↓   ↓
```
"""
function _tensor_halfR(
        Z::HalfBondEnv3site, Ms::Vector{T}, ::Val{1}
    ) where {T <: GenericMPSTensor}
    return @tensoropt hRa[χ dm db; Dw0 Dw′0] :=
        Z[χ; Dw0 De0 Ds0 Dn0] * Ms[2][Dw′0 dm De0 Ds0; Dn′0] * Ms[3][Dn′0 db; Dn0]
end

function _tensor_S(
        hN::AbstractTensor{E, S, 4}, hR::AbstractTensorMap{E, S, 3, 2}, ::Val{1}
    ) where {E, S}
    return @tensor Sa[Dw1 da; Dw′1] :=
        hN[χ da dm db] * conj(hR[χ dm db; Dw1 Dw′1])
end

"""
$(SIGNATURES)

Construct the tensor
```
    ┌-------┐
    | ┌---┬-Z-┬---┐
    | |    ╲ ╱    |
    | └-a--   --b-┘
    ↓   ↓       ↓
```
"""
function _tensor_halfR(
        Z::HalfBondEnv3site, Ms::Vector{T}, ::Val{2}
    ) where {T <: GenericMPSTensor}
    return @tensoropt hRm[χ da db; Dw′0 De0 Ds0 Dn′0] :=
        Z[χ; Dw0 De0 Ds0 Dn0] * Ms[1][Dw0 da; Dw′0] * Ms[3][Dn′0 db; Dn0]
end

function _tensor_S(
        hN::AbstractTensor{E, S, 4}, hR::AbstractTensorMap{E, S, 3, 4}, ::Val{2}
    ) where {E, S}
    return @tensor Sa[Dw′1 dm De1 Ds1; Dn′1] :=
        hN[χ da dm db] * conj(hR[χ da db; Dw′1 De1 Ds1 Dn′1])
end

"""
$(SIGNATURES)

Construct the tensor
```
    ┌-------┐
    | ┌---┬-Z-┬---┐
    | |    ╲ ╱    |
    | └-a---m--   ┘
    ↓   ↓   ↓
```
"""
function _tensor_halfR(
        Z::HalfBondEnv3site, Ms::Vector{T}, ::Val{3}
    ) where {T <: GenericMPSTensor}
    return @tensoropt hRb[χ da dm; Dn′0 Dn0] :=
        Z[χ; Dw0 De0 Ds0 Dn0] * Ms[1][Dw0 da; Dw′0] * Ms[2][Dw′0 dm De0 Ds0; Dn′0]
end

function _tensor_S(
        hN::AbstractTensor{E, S, 4}, hR::AbstractTensorMap{E, S, 3, 2}, ::Val{3}
    ) where {E, S}
    return @tensor Sa[Dn′1 db; Dn1] :=
        hN[χ da dm db] * conj(hR[χ da dm; Dn′1 Dn1])
end

"""
$(SIGNATURES)

Construct half of the norm network
```
    ┌-------┐
    | ┌---┬-Z-┬---┐
    | |    ╲ ╱    |
    | └-a---m---b-┘
    ↓   ↓   ↓   ↓
```
"""
function _tensor_halfN(
        Z::HalfBondEnv3site, Ms::Vector{T}
    ) where {T <: GenericMPSTensor}
    return @tensoropt hN[χ da dm db] :=
        Z[χ; Dw0 De0 Ds0 Dn0] * Ms[1][Dw0 da; Dw′0] *
        Ms[2][Dw′0 dm De0 Ds0; Dn′0] * Ms[3][Dn′0 db; Dn0]
end
function _tensor_halfN(
        hRa::AbstractTensorMap{E, S, 3, 2},
        a::GenericMPSTensor{S, 2}
    ) where {E, S}
    return @tensor hN[χ da dm db] :=
        hRa[χ dm db; Dw0 Dw′0] * a[Dw0 da; Dw′0]
end

"""
$(SIGNATURES)

Calculate the inner product
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├---a2--m2--b2--┤
    |   ↓   ↓   ↓   |
    ├---ā1--m̄1--b̄1--┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
"""
function inner_prod(
        benv::BondEnv3site, Ms1::Vector{T}, Ms2::Vector{T}
    ) where {T <: GenericMPSTensor}
    @assert length(Ms1) == length(Ms2) == 3
    return @tensor benv[Dw1 De1 Ds1 Dn1; Dw0 De0 Ds0 Dn0] *
        conj(Ms1[1][Dw1 da; Dw′1]) *
        conj(Ms1[2][Dw′1 dm De1 Ds1; Dn′1]) * conj(Ms1[3][Dn′1 db; Dn1]) *
        Ms2[1][Dw0 da; Dw′0] * Ms2[2][Dw′0 dm De0 Ds0; Dn′0] * Ms2[3][Dn′0 db; Dn0]
end

function cost_function_als3(
        hN1::AbstractTensor{E, S, 4}, hN2::AbstractTensor{E, S, 4}
    ) where {E, S}
    b12 = only((hN1' * hN2).data)
    b11 = only((hN1' * hN1).data)
    b22 = only((hN2' * hN2).data)
    cost = real(b11) + real(b22) - 2 * real(b12)
    fid = abs2(b12) / abs(b11 * b22)
    return cost, fid
end
