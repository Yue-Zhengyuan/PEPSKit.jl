"""
Algorithm struct for neighborhood tensor update (NTU) of infinite PEPS.
Each NTU run stops when energy starts to increase.
"""
@kwdef struct NTUpdate
    dt::Float64
    maxiter::Int
    # maximum weight difference for convergence
    tol::Float64
    # algorithm to construct bond environment (metric)
    bondenv_alg::NTUEnvAlgorithm
    fixgauge::Bool = true
    # bond truncation after applying time evolution gate
    opt_alg::Union{ALSTruncation,FullEnvTruncation}
    # monitor energy every `ctm_int` steps
    ctm_int::Int = 10
    # CTMRG algorithm to monitor energy
    ctm_alg::CTMRGAlgorithm
end

"""
Neighborhood tensor update for the bond between sites `[row, col]` and `[row, col+1]`.
"""
function _ntu_xbond!(
    row::Int,
    col::Int,
    gate::AbstractTensorMap{T,S,2,2},
    peps::InfiniteWeightPEPS,
    alg::NTUpdate,
) where {T<:Number,S<:ElementarySpace}
    Nr, Nc = size(peps)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    cp1 = _next(col, Nc)
    A, B = peps.vertices[row, col], peps.vertices[row, cp1]
    _alltrue = ntuple(Returns(true), 4)
    A = _absorb_weights(A, peps.weights, row, col, Tuple(1:4), _alltrue, false)
    B = _absorb_weights(B, peps.weights, row, cp1, Tuple(1:4), _alltrue, false)
    X, a, b, Y = _qr_bond(A, B)
    benv = bondenv_ntu(row, col, X, Y, peps, alg.bondenv_alg)
    if alg.fixgauge
        Z = positive_approx(benv)
        Z, a, b, (Linv, Rinv) = fixgauge_benv(Z, a, b)
        X, Y = _fixgauge_benvXY(X, Y, Linv, Rinv)
        benv = Z' * Z
    end
    # apply gate
    a, s, b, = _apply_gate(a, b, gate, truncerr(1e-15))
    a, b = absorb_s(a, s, b)
    # optimize a, b
    a, s, b, info = bond_truncate(a, b, benv, alg.opt_alg)
    A, B = _qr_bond_undo(X, a, b, Y)
    # remove bond weights
    _alltrue = _alltrue[1:3]
    A = _absorb_weights(A, peps.weights, row, col, (NORTH, SOUTH, WEST), _alltrue, true)
    B = _absorb_weights(B, peps.weights, row, cp1, (NORTH, SOUTH, EAST), _alltrue, true)
    peps.vertices[row, col] = A * (100.0 / norm(A, Inf))
    peps.vertices[row, cp1] = B * (100.0 / norm(B, Inf))
    peps.weights[1, row, col] = s / norm(s, Inf)
    return info
end

function _ntu_xbonds!(
    gate::LocalOperator, peps::InfiniteWeightPEPS, alg::NTUpdate; bipartite::Bool
)
    if bipartite
        @assert size(peps) == (2, 2)
        for r in 1:2
            rp1 = _next(r, 2)
            term = get_gateterm(gate, (CartesianIndex(r, 1), CartesianIndex(r, 2)))
            info = _ntu_xbond!(r, 1, term, peps, alg)
            peps.vertices[rp1, 2] = deepcopy(peps.vertices[r, 1])
            peps.vertices[rp1, 1] = deepcopy(peps.vertices[r, 2])
            peps.weights[1, rp1, 2] = deepcopy(peps.weights[1, r, 1])
        end
    else
        for site in CartesianIndices(peps.vertices)
            r, c = Tuple(site)
            term = get_gateterm(gate, (CartesianIndex(r, c), CartesianIndex(r, c + 1)))
            info = _ntu_xbond!(r, c, term, peps, alg)
        end
    end
    return nothing
end

"""
    ntu_iter(gate::LocalOperator, peps::InfiniteWeightPEPS, alg::NTUpdate; bipartite::Bool=false)

One round of neighborhood tensor update on `peps` applying the nearest neighbor `gate`.

Reference: 
- Physical Review B 104, 094411 (2021)
- Physical Review B 106, 195105 (2022)
"""
function ntu_iter(
    gate::LocalOperator, peps::InfiniteWeightPEPS, alg::NTUpdate; bipartite::Bool=false
)
    @assert size(gate.lattice) == size(peps)
    Nr, Nc = size(peps)
    if bipartite
        @assert Nr == Nc == 2
    end
    peps2 = deepcopy(peps)
    gate2 = deepcopy(gate)
    for i in 1:4
        _ntu_xbonds!(gate2, peps2, alg; bipartite)
        peps2 = rotl90(peps2)
        gate2 = rotl90(gate2)
    end
    # for fermions, undo the twists caused by repeated flipping
    for i in CartesianIndices(peps2.vertices)
        twist!(peps2.vertices[i], Tuple(2:5))
    end
    return peps2
end

"""
Perform NTU on InfiniteWeightPEPS with nearest neighbor Hamiltonian `ham`. 

If `bipartite == true` (for square lattice), a unit cell size of `(2, 2)` is assumed, 
as well as tensors and x/y weights which are the same across the diagonals, i.e. at
`(row, col)` and `(row+1, col+1)`.
"""
function ntupdate(
    peps::InfiniteWeightPEPS,
    env::CTMRGEnv,
    ham::LocalOperator,
    alg::NTUpdate,
    ctm_alg::CTMRGAlgorithm;
    bipartite::Bool=false,
)
    time_start = time()
    Nr, Nc = size(peps)
    @info @sprintf(
        "%-4s %7s%10s%12s%11s  %s/%s\n",
        "step",
        "dt",
        "energy",
        "Δe",
        "|Δλ|",
        "speed",
        "meas(s)"
    )
    gate = get_expham(alg.dt, ham)
    wts0, peps0, env0 = deepcopy(peps.weights), deepcopy(env), deepcopy(env)
    energy0, energy, ediff, wtdiff = Inf, 0.0, 0.0, 1.0
    for count in 1:(alg.maxiter)
        time0 = time()
        peps = ntu_iter(gate, peps, alg; bipartite)
        wtdiff = compare_weights(peps.weights, wts0)
        converge = (wtdiff < alg.tol)
        cancel = (count == alg.maxiter)
        wts0 = deepcopy(peps.weights)
        time1 = time()
        if count == 1 || count % alg.ctm_int == 0 || converge || cancel
            # monitor change of energy
            meast0 = time()
            peps_ = InfinitePEPS(peps)
            normalize!.(peps_.A, Inf)
            env, = leading_boundary(env, peps_, alg.ctm_alg)
            energy = cost_function(peps_, env, ham) / (Nr * Nc)
            ediff = energy - energy0
            meast1 = time()
            message = @sprintf(
                "%-4d %7.0e%10.5f%12.3e%11.3e  %.3f/%.3f\n",
                count,
                alg.dt,
                energy,
                ediff,
                wtdiff,
                time1 - time0,
                meast1 - meast0
            )
            cancel ? (@warn message) : (@info message)
            if ediff > 0
                @info "Energy starts to increase. Abort evolution.\n"
                # restore last checkpoint
                peps, env, energy = deepcopy(peps0), deepcopy(env0), energy0
                break
            end
            peps0, env0, energy0 = deepcopy(peps), deepcopy(env), energy
            converge && break
        end
    end
    # reconverge the environment tensors
    for io in (stdout, stderr)
        @printf(io, "Reconverging final env ... \n")
    end
    peps_ = InfinitePEPS(peps)
    normalize!.(peps_.A, Inf)
    env, = leading_boundary(env, peps_, ctm_alg)
    energy = cost_function(peps_, env, ham) / (Nr * Nc)
    time_end = time()
    @printf("Evolution time: %.3f s\n\n", time_end - time_start)
    print(stderr, "\n----------\n\n")
    return peps, env, (; energy, ediff, wtdiff)
end
