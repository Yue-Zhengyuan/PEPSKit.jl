function _get_cluster_permute(
        state::InfiniteState, sites::Vector{CartesianIndex{2}}
    )
    Ms, _, perms = _get_cluster(state, sites)
    Np = (state isa InfinitePEPS) ? Val(1) : Val(2)
    invperms = map(p -> _inv_mpo_perm(p, Np), perms)
    return _permute_cluster(Ms, perms), invperms
end

"""
Neighbourhood tensor update with N-site MPO `gate` (N ≥ 2).
"""
function _ntu_iter(
        state::InfiniteState, gate::Vector{T}, wts::SUWeight,
        sites::Vector{CartesianIndex{2}}, alg::NeighbourUpdate
    ) where {T <: AbstractTensorMap}
    state, wts = copy(state), deepcopy(wts)

    # apply gate MPO without truncation
    Ms, invperms = _get_cluster_permute(state, sites)
    flips = [isdual(space(M, 1)) for M in Iterators.drop(Ms, 1)]
    _flip_virtuals!(Ms, flips) # flip virtual arrows in `Ms` to ←
    _apply_gatempo!(Ms, gate)
    for (M, s, invperm) in zip(Ms, sites, invperms)
        state[s] = permute(M, invperm)
    end

    # truncated Vidal gauge projectors
    truncs = _get_cluster_trunc(alg.opt_alg.trunc, sites)
    Pas, Pbs = _get_allprojs(Ms, truncs)
    _flip_virtuals!(Pas, Pbs, flips) # restore virtual arrows in `Ms`

    # truncate each bond sequentially along the path
    info = (; fid = 1.0)
    nbond = length(sites) - 1
    for (i, (site1, site2)) in enumerate(zip(sites, Iterators.drop(sites, 1)))
        trunc = truncs[i]
        alg′ = (@set alg.opt_alg.trunc = trunc)
        stype1 = (i == 1) ? :first : :middle
        stype2 = (i == nbond) ? :last : :middle
        state, wts, info′ = _bond_truncate(
            state, wts, site1, site2, Pas[i], Pbs[i], (stype1, stype2), alg′
        )
        # record the worst fidelity
        (info′.fid < info.fid) && (info = info′)
    end
    return state, wts, info
end

function _bond_tensor_first(
        A, Pa::MPSBondTensor; gate_ax::Int = 1, kwargs...
    )
    a, X = bond_tensor_first(A; gate_ax, kwargs...)
    @tensor a[-1 -2; -3] := a[-1 -2; 1] * Pa[1; -3]
    return a, X
end

function _bond_tensor_last(
        B, Pb::MPSBondTensor; gate_ax::Int = 1, kwargs...
    )
    b, Y = bond_tensor_last(B; gate_ax, kwargs...)
    @tensor b[-1 -2; -3] := Pb[-1; 1] * b[1 -2; -3]
    return b, Y
end

"""
Truncate a nearest neighbor bond between `site1` and `site2`
after rotating the bond to standard x direction `A ← B`.
"""
function _bond_truncate(
        state::InfiniteState, wts::SUWeight,
        site1::CartesianIndex{2}, site2::CartesianIndex{2},
        Pa::MPSBondTensor, Pb::MPSBondTensor,
        (stype1, stype2)::NTuple{2, Symbol}, alg::NeighbourUpdate
    )
    # rotate bond to standard x direction `A ← B`
    ucell = size(state)[1:2]
    bond, rev = _nn_bondrev(site1, site2)
    dir = first(bond)
    state2 = _bond_rotation(state, dir, rev; inv = false)
    wts2 = _bond_rotation(wts, dir, rev; inv = false)

    # rotated bond tensors
    siteA = _bond_rotation(site1, dir, rev, ucell)
    row, col = siteA[1], siteA[2]
    A, B = state2[row, col], state2[row, col + 1]

    # create bond environment
    qrtrunc = trunctol(; rtol = 1.0e-12)
    a, X = if stype1 == :first
        _bond_tensor_first(A, Pa; trunc = qrtrunc)
    else
        @assert stype1 == :middle
        insertrightunit(Pa, 1), A
    end
    b, Y = if stype2 == :last
        _bond_tensor_last(B, Pb; trunc = qrtrunc)
    else
        @assert stype2 == :middle
        insertrightunit(Pb, 1), B
    end
    benv = bondenv_ntu(row, col, X, Y, state2, alg.bondenv_alg)
    @debug "cond(benv) before gauge fix: $(LinearAlgebra.cond(benv))"
    if alg.fixgauge
        Z = positive_approx(benv)
        Z, a, b, (Linv, Rinv) = fixgauge_benv(Z, a, b)
        X = _fixgauge_benvX(X, Rinv)
        Y = _fixgauge_benvY(Y, Linv)
        benv = Z' * Z
        @debug "cond(L) = $(LinearAlgebra.cond(Linv)); cond(R): $(LinearAlgebra.cond(Rinv))"
        @debug "cond(benv) after gauge fix: $(LinearAlgebra.cond(benv))"
    end

    # truncate
    a, s, b, info = bond_truncate(a, b, benv, alg.opt_alg)

    A = if stype1 == :first
        undo_bond_tensor_first(a, X)
    else
        apply_projector(X, removeunit(a, 2))
    end
    B = if stype2 == :last
        undo_bond_tensor_last(b, Y)
    else
        apply_projector(removeunit(b, 2), Y)
    end

    state2[row, col] = normalize!(A, Inf)
    state2[row, col + 1] = normalize!(B, Inf)
    wts2[1, row, col] = normalize!(s, Inf)

    # rotate back tensors and bond weight
    state2 = _bond_rotation(state2, dir, rev; inv = true)
    wts2 = _bond_rotation(wts2, dir, rev; inv = true)
    return state2, wts2, info
end
