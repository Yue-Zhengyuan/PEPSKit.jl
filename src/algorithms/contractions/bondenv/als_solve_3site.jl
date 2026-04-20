_tensor_R(benv, Ms, x::Int) = _tensor_R(benv, Ms, Val(x))
_tensor_S(benv, Ms, Ms2, x::Int) = _tensor_S(benv, Ms, Ms2, Val(x))

"""
$(SIGNATURES)

Construct the tensor
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├--   --m---b---┤
    |       ↓   ↓   |
    ├--   --m̄---b̄---┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
"""
function _tensor_R(benv::BondEnv3site, Ms::Vector{T}, ::Val{1}) where {T <: GenericMPSTensor}
    (_, m, b) = Ms
    return @tensoropt Ra[Dw1 Dw′1; Dw0 Dw′0] :=
        benv[Dw1 De1 Ds1 Dn1; Dw0 De0 Ds0 Dn0] *
        conj(m[Dw′1 dm De1 Ds1; Dn′1]) * conj(b[Dn′1 db; Dn1]) *
        m[Dw′0 dm De0 Ds0; Dn′0] * b[Dn′0 db; Dn0]
end

"""
$(SIGNATURES)

Construct the tensor
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├---a2--m2--b2--┤
    |   ↓   ↓   ↓   |
    ├--   --m̄---b̄---┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
"""
function _tensor_S(
        benv::BondEnv3site, Ms::Vector{T}, Ms2::Vector{T}, ::Val{1}
    ) where {T <: GenericMPSTensor}
    (_, m, b) = Ms
    (a2, m2, b2) = Ms2
    return @tensoropt Sa[Dw1 da; Dw′1] :=
        benv[Dw1 De1 Ds1 Dn1; Dw0 De0 Ds0 Dn0] *
        conj(m[Dw′1 dm De1 Ds1; Dn′1]) * conj(b[Dn′1 db; Dn1]) *
        a2[Dw0 da; Dw′0] * m2[Dw′0 dm De0 Ds0; Dn′0] * b2[Dn′0 db; Dn0]
end

"""
$(SIGNATURES)

Construct the tensor
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├---a--   --b---┤
    |   ↓       ↓   |
    ├---ā--   --b̄---┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
"""
function _tensor_R(benv::BondEnv3site, Ms::Vector{T}, ::Val{2}) where {T <: GenericMPSTensor}
    (a, _, b) = Ms
    return @tensoropt Rm[Dw′1 De1 Ds1 Dn′1; Dw′0 De0 Ds0 Dn′0] :=
        benv[Dw1 De1 Ds1 Dn1; Dw0 De0 Ds0 Dn0] *
        conj(a[Dw1 da; Dw′1]) * conj(b[Dn′1 db; Dn1]) *
        a[Dw0 da; Dw′0] * b[Dn′0 db; Dn0]
end

"""
$(SIGNATURES)

Construct the tensor
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├---a2--m2--b2--┤
    |   ↓   ↓   ↓   |
    ├---ā--   --b̄---┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
"""
function _tensor_S(
        benv::BondEnv3site, Ms::Vector{T}, Ms2::Vector{T}, ::Val{2}
    ) where {T <: GenericMPSTensor}
    (a, _, b) = Ms
    (a2, m2, b2) = Ms2
    return @tensoropt Sm[Dw′1 dm De1 Ds1; Dn′1] :=
        benv[Dw1 De1 Ds1 Dn1; Dw0 De0 Ds0 Dn0] *
        conj(a[Dw1 da; Dw′1]) * conj(b[Dn′1 db; Dn1]) *
        a2[Dw0 da; Dw′0] * m2[Dw′0 dm De0 Ds0; Dn′0] * b2[Dn′0 db; Dn0]
end

"""
$(SIGNATURES)

Construct the tensor
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├---a---m--   --┤
    |   ↓   ↓       |
    ├---ā---m̄--   --┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
"""
function _tensor_R(benv::BondEnv3site, Ms::Vector{T}, ::Val{3}) where {T <: GenericMPSTensor}
    (a, m, _) = Ms
    return @tensoropt Rb[Dn′1 Dn1; Dn′0 Dn0] :=
        benv[Dw1 De1 Ds1 Dn1; Dw0 De0 Ds0 Dn0] *
        conj(a[Dw1 da; Dw′1]) * conj(m[Dw′1 dm De1 Ds1; Dn′1]) *
        a[Dw0 da; Dw′0] * m[Dw′0 dm De0 Ds0; Dn′0]
end

"""
$(SIGNATURES)

Construct the tensor
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├---a2--m2--b2--┤
    |   ↓   ↓   ↓   |
    ├---ā---m̄--   --┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
"""
function _tensor_S(
        benv::BondEnv3site, Ms::Vector{T}, Ms2::Vector{T}, ::Val{3}
    ) where {T <: GenericMPSTensor}
    (a, m, _) = Ms
    a2, m2, b2 = Ms2
    return @tensoropt Rb[Dn′1 db; Dn1] :=
        benv[Dw1 De1 Ds1 Dn1; Dw0 De0 Ds0 Dn0] *
        conj(a[Dw1 da; Dw′1]) * conj(m[Dw′1 dm De1 Ds1; Dn′1]) *
        a2[Dw0 da; Dw′0] * m2[Dw′0 dm De0 Ds0; Dn′0] * b2[Dn′0 db; Dn0]
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
    (a1, m1, b1) = Ms1
    (a2, m2, b2) = Ms2
    return @tensor contractcheck = true benv[Dw1 De1 Ds1 Dn1; Dw0 De0 Ds0 Dn0] *
        conj(a1[Dw1 da; Dw′1]) *
        conj(m1[Dw′1 dm De1 Ds1; Dn′1]) * conj(b1[Dn′1 db; Dn1]) *
        a2[Dw0 da; Dw′0] * m2[Dw′0 dm De0 Ds0; Dn′0] * b2[Dn′0 db; Dn0]
end
