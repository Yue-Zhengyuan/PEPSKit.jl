"""
Truncate bonds in the 3-site cluster `Ms = [a, m, b]` using
the environment (norm tensor) `benv` surrounding `Ms` as
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├---a===m===b---┤
    |   ↓   ↓   ↓   |
    ├---ā===m̄===b̄---┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
`m` is the tensor at the middle site, while `a`, `b` are
reduced bond tensor from the first and the last site.

Input tensors `Ms = [a, m, b]` have MPS axis order along
the southeast 3-site cluster.

Reference: Phys. Rev. B 97, 174408 (2018)
"""
function se3site_truncate(
        Ms::Vector{T}, benv::BondEnv3site, alg::ALSTruncation
    ) where {T <: GenericMPSTensor}
    # dual check
    @assert length(Ms) == 3
    @assert codomain(benv) == domain(benv)
    time00 = time()
    verbose = (alg.check_interval > 0)
    # standardize virtual arrows
    flips = [isdual(space(M, 1)) for M in Ms[2:end]]
    Ms_trunc = deepcopy(Ms)
    _flip_virtuals!(Ms_trunc, flips)
    # initialize truncated a, m, b
    wts0, = _cluster_truncate!(Ms_trunc, fill(alg.trunc, 2))
    # initial cost and fidelity
    cost00, fid = cost_function_als(benv, Ms_trunc, Ms)
    cost0, fid0, Δcost, Δfid, Δs = cost00, fid, NaN, NaN, NaN
    verbose && @info "ALS3 init" * _als_message(0, cost0, fid, Δcost, Δfid, Δs, 0.0)
    for iter in 1:(alg.maxiter)
        time0 = time()
        for (idx, x0) in enumerate(Ms_trunc)
            R = _tensor_R(benv, Ms_trunc, idx)
            S = _tensor_S(benv, Ms_trunc, Ms, idx)
            Ms_trunc[idx] = _solve_als_pinv(R, S)
        end
        # compare cost, fidelity, bond weights
        cost, fid = cost_function_als(benv, Ms_trunc, Ms)
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
