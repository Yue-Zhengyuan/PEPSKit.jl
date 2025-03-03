"""
Algorithm struct for full update (FU) of infinite PEPS.
Each FU run stops when the energy starts to increase.
"""
@kwdef struct FullUpdate
    dt::Float64
    maxiter::Int
    fixgauge::Bool = true
    # bond truncation after applying time evolution gate
    opt_alg::Union{ALSTruncation,FullEnvTruncation}
    # SequentialCTMRG column move after updating a column of bonds
    colmove_alg::SequentialCTMRG
    # interval to reconverge environments
    reconv_int::Int = 10
    # CTMRG for reconverging environment
    reconv_alg::CTMRGAlgorithm
end

"""
Full update for the bond between `[row, col]` and `[row, col+1]`.
"""
function _fu_bondx!(
    row::Int,
    col::Int,
    gate::AbstractTensorMap{T,S,2,2},
    peps::InfinitePEPS,
    env::CTMRGEnv,
    alg::FullUpdate,
) where {T<:Number,S<:ElementarySpace}
    Nr, Nc = size(peps)
    cp1 = _next(col, Nc)
    A, B = peps[row, col], peps[row, cp1]
    X, a, b, Y = _qr_bond(A, B)
    # positive/negative-definite approximant: benv = ± Z Z†
    benv = bondenv_fu(row, col, X, Y, env)
    Z = positive_approx(benv)
    # fix gauge
    if alg.fixgauge
        Z, a, b, (Linv, Rinv) = fixgauge_benv(Z, a, b)
        X, Y = _fixgauge_benvXY(X, Y, Linv, Rinv)
    end
    benv = Z' * Z
    # apply gate
    a, s, b, = _apply_gate(a, b, gate, truncerr(1e-15))
    a, b = absorb_s(a, s, b)
    # optimize a, b
    a, s, b, info = bond_truncate(a, b, benv, alg.opt_alg)
    a, b = absorb_s(a, s, b)
    a /= norm(a, Inf)
    b /= norm(b, Inf)
    A, B = _qr_bond_undo(X, a, b, Y)
    peps.A[row, col] = A / norm(A, Inf)
    peps.A[row, cp1] = B / norm(B, Inf)
    return s, info
end

"""
Update all horizontal bonds in the c-th column
(i.e. `(r,c) (r,c+1)` for all `r = 1, ..., Nr`).
To update rows, rotate the network clockwise by 90 degrees.
"""
function _update_column!(
    col::Int, gate::LocalOperator, peps::InfinitePEPS, env::CTMRGEnv, alg::FullUpdate
)
    Nr, Nc = size(peps)
    @assert 1 <= col <= Nc
    fid = 1.0
    wts_col = Vector{PEPSWeight}(undef, Nr)
    for row in 1:Nr
        term = get_gateterm(gate, (CartesianIndex(row, col), CartesianIndex(row, col + 1)))
        wts_col[row], info = _fu_bondx!(row, col, term, peps, env, alg)
        fid = min(fid, info.fid)
    end
    # update CTMRGEnv
    network = InfiniteSquareNetwork(peps)
    env2, info = ctmrg_leftmove(col, network, env, alg.colmove_alg)
    env2, info = ctmrg_rightmove(_next(col, Nc), network, env2, alg.colmove_alg)
    for c in [col, _next(col, Nc)]
        env.corners[:, :, c] = env2.corners[:, :, c]
        env.edges[:, :, c] = env2.edges[:, :, c]
    end
    return wts_col, fid
end

"""
One round of full update on the input InfinitePEPS `peps` and its CTMRGEnv `env`

Reference: Physical Review B 92, 035142 (2015)
"""
function fu_iter(gate::LocalOperator, peps::InfinitePEPS, env::CTMRGEnv, alg::FullUpdate)
    Nr, Nc = size(peps)
    fid = 1.0
    peps2, env2 = deepcopy(peps), deepcopy(env)
    wts = Array{PEPSWeight}(undef, 2, Nr, Nc)
    for col in 1:Nc
        wts[1, :, col], fid_col = _update_column!(col, gate, peps2, env2, alg)
        fid = min(fid, fid_col)
    end
    peps2, env2 = rotr90(peps2), rotr90(env2)
    gate_rotated = rotr90(gate)
    for row in 1:Nr
        # the row-th column after rotr90 was (Nr+1-row)-th row
        wts[2, Nr + 1 - row, :], fid_row = _update_column!(
            row, gate_rotated, peps2, env2, alg
        )
        fid = min(fid, fid_row)
    end
    peps2, env2 = rotl90(peps2), rotl90(env2)
    return peps2, env2, SUWeight(wts), fid
end

"""
Perform full update with nearest neighbor Hamiltonian `ham`.
After FU stops, the final environment is calculated with CTMRG algorithm `ctm_alg`.
"""
function fullupdate(
    peps::InfinitePEPS,
    env::CTMRGEnv,
    ham::LocalOperator,
    fu_alg::FullUpdate,
    ctm_alg::CTMRGAlgorithm,
)
    time_start = time()
    Nr, Nc = size(peps)
    @printf(
        "%-4s %7s%10s%12s%11s  %s/%s\n",
        "step",
        "dt",
        "energy",
        "Δe",
        "|Δλ|",
        "speed",
        "meas(s)"
    )
    gate = get_gate(fu_alg.dt, ham)
    peps0, env0, wts0, wts = deepcopy(peps), deepcopy(env), nothing, nothing
    energy0, energy, ediff, wtdiff = Inf, 0.0, 0.0, NaN
    for count in 1:(fu_alg.maxiter)
        time0 = time()
        peps, env, wts, fid = fu_iter(gate, peps, env, fu_alg)
        time1 = time()
        if count == 1 || count % fu_alg.reconv_int == 0
            # reconverge environment
            meast0 = time()
            println(stderr, "---- FU step $count: reconverging env ----")
            env, = leading_boundary(env, peps, fu_alg.reconv_alg)
            energy = cost_function(peps, env, ham) / (Nr * Nc)
            meast1 = time()
            ediff = energy - energy0
            wtdiff = (count == 1) ? NaN : compare_weights(wts, wts0)
            @printf(
                "%-4d %7.0e%10.5f%12.3e%11.3e  %.3f/%.3f\n",
                count,
                fu_alg.dt,
                energy,
                ediff,
                wtdiff,
                time1 - time0,
                meast1 - meast0
            )
            if ediff > 0
                @printf("Energy starts to increase. Abort evolution.\n")
                # restore last checkpoint
                peps, env, energy, wts = deepcopy(peps0),
                deepcopy(env0), energy0,
                deepcopy(wts0)
                break
            end
            peps0, env0, energy0, wts0 = deepcopy(peps),
            deepcopy(env), energy,
            deepcopy(wts)
        end
    end
    # reconverge the environment tensors
    for io in (stdout, stderr)
        @printf(io, "Reconverging final env ... \n")
    end
    env, = leading_boundary(env, peps, ctm_alg)
    energy = cost_function(peps, env, ham) / (Nr * Nc)
    time_end = time()
    @printf("Evolution time: %.3f s\n\n", time_end - time_start)
    print(stderr, "\n----------\n\n")
    return peps, env, (; energy, ediff, wtdiff)
end
