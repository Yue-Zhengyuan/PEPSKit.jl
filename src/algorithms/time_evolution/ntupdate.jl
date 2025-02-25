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
function _ntu_bondx!(
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
    A = _absorb_weight(A, row, col, "", peps.weights)
    B = _absorb_weight(B, row, cp1, "", peps.weights)
    X, a, b, Y = _qr_bond(A, B)
    benv = bondenv_ntu(row, col, X, Y, peps, alg.bondenv_alg)
    # apply gate
    a, s, b, = _apply_gate(a, b, gate, truncerr(1e-15))
    a, b = absorb_s(a, s, b)
    # optimize a, b
    a, s, b, (cost, fid) = bond_truncate(a, b, benv, alg.opt_alg)
    A, B = _qr_bond_undo(X, a, b, Y)
    # remove bond weights
    for ax in (2, 4, 5)
        A = absorb_weight(A, row, col, ax, peps.weights; sqrtwt=true, invwt=true)
    end
    for ax in (2, 3, 4)
        B = absorb_weight(B, row, cp1, ax, peps.weights; sqrtwt=true, invwt=true)
    end
    peps.vertices[row, col] = A / norm(A, Inf)
    peps.vertices[row, cp1] = B / norm(B, Inf)
    peps.weights[1, row, col] = s / norm(s, Inf)
    return cost, fid
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
    # TODO: make algorithm independent on the choice of dual in the network
    for (r, c) in Iterators.product(1:Nr, 1:Nc)
        @assert [isdual(space(peps.vertices[r, c], ax)) for ax in 1:5] == [0, 1, 1, 0, 0]
        @assert [isdual(space(peps.weights[1, r, c], ax)) for ax in 1:2] == [0, 1]
        @assert [isdual(space(peps.weights[2, r, c], ax)) for ax in 1:2] == [0, 1]
    end
    peps2 = deepcopy(peps)
    gate_mirrored = mirror_antidiag(gate)
    for direction in 1:2
        if direction == 2
            peps2 = mirror_antidiag(peps2)
        end
        if bipartite
            for r in 1:2
                rp1 = _next(r, 2)
                term = get_gateterm(
                    direction == 1 ? gate : gate_mirrored,
                    (CartesianIndex(r, 1), CartesianIndex(r, 2)),
                )
                ϵ = _ntu_bondx!(r, 1, term, peps2, alg)
                peps2.vertices[rp1, 2] = deepcopy(peps2.vertices[r, 1])
                peps2.vertices[rp1, 1] = deepcopy(peps2.vertices[r, 2])
                peps2.weights[1, rp1, 2] = deepcopy(peps2.weights[1, r, 1])
            end
        else
            for site in CartesianIndices(peps2.vertices)
                r, c = Tuple(site)
                term = get_gateterm(
                    direction == 1 ? gate : gate_mirrored,
                    (CartesianIndex(r, c), CartesianIndex(r, c + 1)),
                )
                ϵ = _ntu_bondx!(r, c, term, peps2, alg)
            end
        end
        if direction == 2
            peps2 = mirror_antidiag(peps2)
        end
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
    gate = get_gate(alg.dt, ham)
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
    env, = leading_boundary(env, peps_, ctm_alg)
    energy = cost_function(peps_, env, ham) / (Nr * Nc)
    time_end = time()
    @printf("Evolution time: %.3f s\n\n", time_end - time_start)
    print(stderr, "\n----------\n\n")
    return peps, env, (; energy, ediff, wtdiff)
end
