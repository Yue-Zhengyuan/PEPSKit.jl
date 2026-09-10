"""
$(TYPEDEF)

Projector algorithm implementing the `qr` decomposition of a column-enlarged corner.

## Fields

$(TYPEDFIELDS)

## Constructors

    C4vQRProjector(; kwargs...)

Construct the C₄ᵥ `qr`-based projector algorithm based on the following keyword arguments:

* `decomposition_alg=QRAdjoint()` : `left_orth` algorithm including the reverse rule. See [`QRAdjoint`](@ref).
"""
struct C4vQRProjector{S} <: ProjectorAlgorithm
    # TODO: support all `left_orth` algorithms
    decomposition_alg::S
end
function C4vQRProjector(; kwargs...)
    return ProjectorAlgorithm(; alg = :C4vQRProjector, kwargs...)
end
PROJECTOR_SYMBOLS[:C4vQRProjector] = C4vQRProjector

decomposition_algorithm(alg::C4vQRProjector) = alg.decomposition_alg

# no truncation
_set_truncation(alg::C4vQRProjector, ::TruncationStrategy) = alg
_set_decomposition_truncation(alg::C4vQRProjector, ::TruncationStrategy) = alg

"""
Compute the column-enlarged northwest corner for C₄ᵥ QR-CTMRG.
"""
function c4v_enlarge(env, ::C4vQRProjector)
    return TensorMap(ColumnEnlargedCorner(env, (NORTHWEST, 1, 1)))
end

"""
Compute the C₄ᵥ projector by decomposing the column-enlarged corner with `left_orth`.
```
                   R--←--
                   ↓
    C-←-E-←-  =  [~Q~]
    ↓   |        ↓   |
```
"""
function c4v_projector!(enlarged_corner, alg::C4vQRProjector)
    Q, R = left_orth!(enlarged_corner, decomposition_algorithm(alg))
    # TODO: what's a meaningful way to compute a truncation error/condition number in this scheme?
    return Q, (; Q, R, truncation_error = zero(scalartype(Q)))
end

"""
Renormalize the single corner tensor
```
    C-←-E-←-|~~~|
    |   |   | P |-←-
    E---A---|~~~|
    |   |
    [~P']
      ↓
```
Using the already calculated QR decomposition
```
                   R--←--
                   ↓
    C-←-E-←-  =  [~P~]
    ↓   |        ↓   |
```
we rewrite the renormalized corner as
```
    R-←-|~~~|
    ↓   | P |-←-
    E′--|~~~|
    ↓
```
which reuses the renormalized edge `E′` (`new_edge`).
(Credit: https://github.com/qiyang-ustc/QRCTM/blob/dd160116c3d7b02076691ceaf0a9833511ae532d/heisenberg.py#L80)
"""
# TODO: possible missing twists for fermions
function c4v_qr_renormalize_corner(new_edge::CTMRGEdgeTensor, projector, R)
    # contract edge and R
    edge′ = physical_flip(new_edge)
    ER = edge′ * twistdual(R, 1)
    # contract (edge, R) with projector
    new_corner = contract_edges(ER, projector)
    new_corner = project_hermitian(new_corner)
    return new_corner / norm(new_corner)
end

"""Contract two CTMRG edge tensors into a corner tensor."""
@generated function contract_edges(
        EL::CTMRGEdgeTensor{T, S, N}, ER::CTMRGEdgeTensor{T, S, N}
    ) where {T, S, N}
    C´_e = tensorexpr(:C´, -1, -2)
    EL_e = tensorexpr(:EL, (-1, (2:N)...), 1)
    ER_e = tensorexpr(:ER, 1:N, -2)
    return macroexpand(@__MODULE__, :(return @tensor $C´_e := $EL_e * $ER_e))
end
