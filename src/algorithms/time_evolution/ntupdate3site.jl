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
    state2, wts2 = copy(state), deepcopy(wts)

    # apply gate MPO
    Ms, invperms = _get_cluster_permute(state2, sites)
    flips = [isdual(space(M, 1)) for M in Iterators.drop(Ms, 1)]
    _flip_virtuals!(Ms, flips) # flip virtual arrows in `Ms` to ←
    _apply_gatempo!(Ms, gate)

    # put un-truncated tensors in `state2`
    # arrow direction is temporarily changed
    for (M, s, invperm) in zip(Ms, sites, invperms)
        state2[s] = permute(M, invperm)
    end
    
    # Vidal gauge projectors
    truncs = _get_cluster_trunc(alg.opt_alg.trunc, sites)
    Pas, Pbs = _get_allprojs(Ms, truncs)
    _flip_virtuals!(Pas, Pbs, flips)

    # bond-wise truncation
    # TODO: reduce code duplication
    fid = 1.0
    for (i, (siteA, siteB)) in enumerate(zip(sites, Iterators.drop(sites, 1)))
        # rotate to standard x direction `A ← B`
        bond, rev = _nn_bondrev(siteA, siteB)
        dir = first(bond)
        state2 = _bond_rotation(state2, dir, rev; inv = false)
        wts2 = _bond_rotation(wts2, dir, rev; inv = false)

        # rotated bond tensors
        ucell = size(state2)[1:2]
        siteA′ = _bond_rotation(siteA, dir, rev, ucell)
        row, col = siteA′[1], siteA′[2]
        A, B = state2[row, col], state2[row, col + 1]

        # apply projectors on current bond
        # (also restoring arrow direction)
        Pa, Pb = Pas[i], Pbs[i]
        A = apply_projector(A, Pa)
        B = apply_projector(Pb, B)

        # factor out bond tensor
        a, X = bond_tensor_first(A; positive = true)
        b, Y = bond_tensor_last(B; positive = true)

        # create bond environment
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

        # truncation
        opt_alg = alg.opt_alg
        @reset opt_alg.trunc = truncs[i]
        a, s, b, info = bond_truncate(a, b, benv, opt_alg)
        # record the worst fidelity
        (info.fid < fid) && (fid = info.fid)

        A = undo_bond_tensor_first(a, X)
        B = undo_bond_tensor_last(b, Y)
        state2[row, col] = normalize!(A, Inf)
        state2[row, col + 1] = normalize!(B, Inf)
        wts2[1, row, col] = normalize!(s, Inf)

        # rotate back tensors and bond weight
        # TODO: only restore orientation after truncating all bonds
        state2 = _bond_rotation(state2, dir, rev; inv = true)
        wts2 = _bond_rotation(wts2, dir, rev; inv = true)
    end

    return state2, wts2, (; fid = 1.0)
end
