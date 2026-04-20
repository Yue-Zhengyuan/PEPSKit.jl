"""
Truncate bonds in the 3-site cluster `Ms = [a, m, b]` using
the environment (norm tensor) `Z' * Z` surrounding `Ms` as
```
    ┌-------┐
    | ┌---┬-Z-┬---┐
    | |    ╲ ╱    |
    | └-a===m===b-┘
    ↓   ↓   ↓   ↓
    | ┌-ā===m̄===b̄-┐
    | |    ╱ ╲    |
    | └---┴-Z̄-┴---┘
    └-------┘
```
`m` is the tensor at the middle site, while `a`, `b` are
reduced bond tensor from the first and the last site.

Input tensors `Ms = [a, m, b]` have MPS axis order along
the southeast 3-site cluster.

Reference: Phys. Rev. B 97, 174408 (2018)
"""
function se3site_truncate(
        Ms::Vector{T}, Z::HalfBondEnv3site, alg::ALSTruncation
    ) where {T <: GenericMPSTensor}
    # dual check
    @assert length(Ms) == 3
    time00 = time()
    verbose = (alg.check_interval > 0)

    # initialize truncated a, m, b
    flips = [isdual(space(M, 1)) for M in Ms[2:end]]
    Ms_trunc = deepcopy(Ms)
    _flip_virtuals!(Ms_trunc, flips)
    wts0, = _cluster_truncate!(Ms_trunc, fill(alg.trunc, 2))

    # cache of half R tensors
    hR_cache = [_tensor_halfR(Z, Ms_trunc, idx) for idx in 1:3]

    # half of the norm network
    hN = _tensor_halfN(hR_cache[1], Ms_trunc[1])
    hN2 = _tensor_halfN(Z, Ms)

    # initial cost and fidelity
    cost00, fid = cost_function_als3(hN, hN2)
    cost0, fid0, Δcost, Δfid, Δs = cost00, fid, NaN, NaN, NaN
    verbose && @info "ALS3 init" * _als_message(0, cost0, fid, Δcost, Δfid, Δs, 0.0)

    for iter in 1:(alg.maxiter)
        time0 = time()
        for (idx, x0) in enumerate(Ms_trunc)
            hR = hR_cache[idx]
            R = hR' * hR
            S = _tensor_S(hN2, hR, idx)
            Ms_trunc[idx] = _solve_als_pinv(R, S)
            # update cache needed by optimization of next site
            idx_next = _next(idx, 3)
            hR_cache[idx_next] = _tensor_halfR(Z, Ms_trunc, idx_next)
        end
        # compare cost, fidelity, bond weights
        hN = _tensor_halfN(hR_cache[1], Ms_trunc[1])
        cost, fid = cost_function_als3(hN, hN2)
        wts = _get_allprojs(Ms_trunc, fill(notrunc(), 2))[3]
        Δcost = abs(cost - cost0) / cost00
        Δfid = abs(fid - fid0)
        Δs = mean(
            _singular_value_distance(s, s0) for (s, s0) in zip(wts, wts0)
        ) / norm(wts0[1], Inf)
        cost0, fid0, wts0 = cost, fid, wts
        time1 = time()
        converge = (Δs < alg.tol)
        cancel = (iter == alg.maxiter)
        showinfo =
            cancel || (verbose && (converge || iter == 1 || iter % alg.check_interval == 0))
        if showinfo
            message = _als_message(
                iter, cost, fid, Δcost, Δfid, Δs,
                time1 - ((cancel || converge) ? time00 : time0),
            )
            if converge
                @info "ALS3 conv" * message
            elseif cancel
                @warn "ALS3 cancel" * message
            else
                @info "ALS3 iter" * message
            end
        end
        converge && break
    end
    # convert to Vidal gauge at the end
    wts, = _cluster_truncate!(Ms_trunc, fill(notrunc(), 2))
    # restore virtual arrows
    _flip_virtuals!(Ms_trunc, flips)
    return Ms_trunc, wts, (; fid, Δfid, Δs)
end
