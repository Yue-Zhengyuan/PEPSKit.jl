@kwdef struct ALS2BondTruncation{T}
    trunc::T
    maxiter::Int = 50
    inneriter::Int = 4
    tol::Float64 = 1.0e-9
    check_interval::Int = 0
end

function se3site_truncate(
        Ms::Vector{T}, benv::BondEnv3site, alg::ALS2BondTruncation
    ) where {T <: GenericMPSTensor}
    # dual check
    @assert length(Ms) == 3
    time00 = time()
    verbose = (alg.check_interval > 0)
    @assert alg.inneriter > 0

    # untruncated things
    ket2 = _combine_ket(Ms...)
    benv_ket2 = _benv_ket(benv, ket2)
    b22 = real(_als_norm(ket2, benv_ket2))

    # initialize truncated bond tensors
    xs, _, wts0, flips = _als3_init_truncate(Ms, alg.trunc)

    # initial cost and fidelity
    cost00, fid = cost_function_als(
        _als_tensor_R(benv, xs, 1),
        _als_tensor_S(benv_ket2, xs, 1), xs[1], b22
    )
    cost0, fid0, Δcost, Δfid, Δs = cost00, fid, NaN, NaN, NaN
    verbose && @info "ALS3 init" * _als_message(0, cost0, fid, Δcost, Δfid, Δs, 0.0)

    for iter in 1:(alg.maxiter)
        time0 = time()

        # optimize on first bond
        p, Y = _bond_tensor_prev(xs[2])
        benv_Yb = _benv_Yb(benv, Y, xs[3])
        benv_ket_Yb = _benv_ket_Yb(benv_ket2, Y, xs[3])
        for _ in 1:alg.inneriter
            # optimize xs[1]
            R = _als_tensor_Ra(benv_Yb, p)
            S = _als_tensor_Sa(benv_ket_Yb, p)
            xs[1] = _solve_als_pinv(R, S)
            # optimize p
            R = _als_tensor_Rb(benv_Yb, xs[1])
            S = _als_tensor_Sb(benv_ket_Yb, xs[1])
            p = _solve_als_pinv(R, S)
        end
        xs[2] = _undo_bond_tensor_prev(p, Y)

        # optimize on second bond
        X, q = _bond_tensor_next(xs[2])
        benv_aX = _benv_aX(benv, xs[1], X)
        benv_ket_aX = _benv_ket_aX(benv_ket2, xs[1], X)
        for _ in 1:alg.inneriter
            # optimize q
            R = _als_tensor_Ra(benv_aX, xs[3])
            S = _als_tensor_Sa(benv_ket_aX, xs[3])
            q = _solve_als_pinv(R, S)
            # optimize xs[3]
            R = _als_tensor_Rb(benv_aX, q)
            S = _als_tensor_Sb(benv_ket_aX, q)
            xs[3] = _solve_als_pinv(R, S)
        end
        xs[2] = _undo_bond_tensor_next(X, q)

        # compare cost, fidelity, bond weights
        cost, fid = cost_function_als(
            _als_tensor_R(benv, xs, 1),
            _als_tensor_S(benv_ket2, xs, 1), xs[1], b22
        )
        wts = _get_allprojs(xs, fill(notrunc(), 2))[3]
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
    wts, = _cluster_truncate!(xs, fill(notrunc(), 2))
    # restore virtual arrows
    _flip_virtuals!(xs, flips)
    return xs, wts, (; fid, Δfid, Δs)
end
