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
    ctmrg_rightmove(col::Int, peps::InfinitePEPS, env::CTMRGEnv, alg::SequentialCTMRG)

CTMRG right-move to update CTMRGEnv in the c-th column
```
    absorb <---
    ←-- T1 ← C2     r-1
        ‖    ↑
    === M' = T2     r
        ‖    ↑
    --→ T3 → C3     r+1
        c   c+1
```
"""
function ctmrg_rightmove(col::Int, peps::InfinitePEPS, env::CTMRGEnv, alg::SequentialCTMRG)
    Nr, Nc = size(peps)
    @assert 1 <= col <= Nc
    env, info = ctmrg_leftmove(Nc + 1 - col, rot180(peps), rot180(env), alg)
    return rot180(env), info
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
    # TODO: relax dual requirement on the bonds
    @assert !isdual(domain(A)[2])
    #= QR and LQ decomposition

        2   1               1             2
        | ↗                 |            ↗
    5 - A ← 3   ====>   4 - X ← 2   1 ← aR ← 3
        |                   |
        4                   3
    =#
    X, aR0 = leftorth(A, ((2, 4, 5), (1, 3)); alg=QRpos())
    X = permute(X, (1, 4, 2, 3))
    aR0 = permute(aR0, (1, 2, 3))
    #=
        2   1                 2         2
        | ↗                 ↗           |
    5 ← B - 3   ====>  1 ← bL → 3   1 → Y - 3
        |                               |
        4                               4
    =#
    Y, bL0 = leftorth(B, ((2, 3, 4), (1, 5)); alg=QRpos())
    Y = permute(Y, (1, 2, 3, 4))
    bL0 = permute(bL0, (3, 2, 1))
    benv = bondenv_fu(row, col, X, Y, env)
    # positive/negative-definite approximant: env = ± Z Z†
    Z = positive_approx(benv)
    # fix gauge
    if alg.fixgauge
        Z, X, Y, aR0, bL0 = fu_fixgauge(Z, X, Y, aR0, bL0)
    end
    benv = Z' * Z
    @assert [isdual(space(benv, ax)) for ax in 1:4] == [0, 0, 1, 1]
    #= apply gate

            -2          -3
            ↑           ↑
            |----gate---|
            ↑           ↑
            1           2
            ↑           ↑
        -1← aR -← 3 -← bL → -4
    =#
    aR2bL2 = ncon((gate, aR0, bL0), ([-2, -3, 1, 2], [-1, 1, 3], [3, 2, -4]))
    # initialize un-truncated tensors using SVD
    aR, s_cut, bL, ϵ = tsvd(
        aR2bL2, ((1, 2), (3, 4)); trunc=truncerr(1e-15), alg=TensorKit.SVD()
    )
    aR, bL = absorb_s(aR, s_cut, bL)
    aR, bL = permute(aR, (1, 2, 3)), permute(bL, (1, 2, 3))
    # optimize aR, bL
    aR, s, bL, (cost, fid) = bond_optimize(aR, bL, benv, alg.opt_alg)
    aR, bL = absorb_s(aR, s, bL)
    aR /= norm(aR, Inf)
    bL /= norm(bL, Inf)
    #= update and normalize peps, ms

            -2        -1               -1     -2
            |        ↗                ↗       |
        -5- X ← 1 ← aR ← -3     -5 ← bL → 1 → Y - -3
            |                                 |
            -4                                -4
    =#
    @tensor A[-1; -2 -3 -4 -5] := X[-2 1 -4 -5] * aR[1 -1 -3]
    @tensor B[-1; -2 -3 -4 -5] := bL[-5 -1 1] * Y[-2 -3 -4 1]
    peps.A[row, col] = A / norm(A, Inf)
    peps.A[row, cp1] = B / norm(B, Inf)
    return s, cost, fid
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
    localfid = 0.0
    costs = zeros(Nr)
    wts_col = Vector{PEPSWeight}(undef, Nr)
    #= Axis order of X, aR, Y, bL

            1             2            2         1
            |            ↗           ↗           |
        4 - X ← 2   1 ← aR ← 3  1 ← bL → 3   4 → Y - 2
            |                                    |
            3                                    3
    =#
    for row in 1:Nr
        term = get_gateterm(gate, (CartesianIndex(row, col), CartesianIndex(row, col + 1)))
        wts_col[row], cost, fid = _fu_bondx!(row, col, term, peps, env, alg)
        costs[row] = cost
        localfid += fid
    end
    # update CTMRGEnv
    env2, info = ctmrg_leftmove(col, peps, env, alg.colmove_alg)
    env2, info = ctmrg_rightmove(_next(col, Nc), peps, env2, alg.colmove_alg)
    for c in [col, _next(col, Nc)]
        env.corners[:, :, c] = env2.corners[:, :, c]
        env.edges[:, :, c] = env2.edges[:, :, c]
    end
    return wts_col, localfid, costs
end

"""
One round of full update on the input InfinitePEPS `peps` and its CTMRGEnv `env`

Reference: Physical Review B 92, 035142 (2015)
"""
function fu_iter(gate::LocalOperator, peps::InfinitePEPS, env::CTMRGEnv, alg::FullUpdate)
    Nr, Nc = size(peps)
    fid, maxcost = 0.0, 0.0
    peps2, env2 = deepcopy(peps), deepcopy(env)
    wts = Array{PEPSWeight}(undef, 2, Nr, Nc)
    for col in 1:Nc
        wts[1, :, col], tmpfid, costs = _update_column!(col, gate, peps2, env2, alg)
        fid += tmpfid
        maxcost = max(maxcost, maximum(costs))
    end
    peps2, env2 = rotr90(peps2), rotr90(env2)
    gate_rotated = rotr90(gate)
    for row in 1:Nr
        # the row-th column after rotr90 was (Nr+1-row)-th row
        wts[2, Nr + 1 - row, :], tmpfid, costs = _update_column!(
            row, gate_rotated, peps2, env2, alg
        )
        fid += tmpfid
        maxcost = max(maxcost, maximum(costs))
    end
    peps2, env2 = rotl90(peps2), rotl90(env2)
    fid /= (2 * Nr * Nc)
    return peps2, env2, SUWeight(wts), (fid, maxcost)
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
        peps, env, wts, (fid, cost) = fu_iter(gate, peps, env, fu_alg)
        wtdiff = (count == 1) ? NaN : compare_weights(wts, wts0)
        wts0 = deepcopy(wts)
        time1 = time()
        if count == 1 || count % fu_alg.reconv_int == 0
            # reconverge environment
            meast0 = time()
            println(stderr, "---- FU step $count: reconverging env ----")
            env, = leading_boundary(env, peps, fu_alg.reconv_alg)
            energy = cost_function(peps, env, ham) / (Nr * Nc)
            meast1 = time()
            ediff = energy - energy0
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
                peps, env, energy = deepcopy(peps0), deepcopy(env0), energy0
                break
            end
            peps0, env0, energy0 = deepcopy(peps), deepcopy(env), energy
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
