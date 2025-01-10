"""
Algorithm struct for neighborhood tensor update (NTU) of infinite PEPS.
Each NTU run stops when energy starts to increase.
"""
@kwdef struct PEPSUpdate
    dt::Float64
    maxiter::Int
    trscheme::TensorKit.TruncationScheme
    bondenv_alg::BondEnvAlgorithm
    als_alg::ALSOptimize = ALSOptimize()
end

function truncation_scheme(alg::PEPSUpdate, v::ElementarySpace)
    if alg.trscheme isa FixedSpaceTruncation
        return truncspace(v)
    else
        return alg.trscheme
    end
end

"""
Neighborhood tensor update of the x-bond between sites `[r,c]` and `[r,c+1]`
"""
function _update_bondx!(
    row::Int, col::Int, gate::AbstractTensorMap{S,2,2}, peps::InfinitePEPS, alg::PEPSUpdate
) where {S<:ElementarySpace}
    Nr, Nc = size(peps)
    #= perform SVD of the gate

        ↑       ↑       ↑       ↑
        |--gate-| ====> Ta -←- Tb
        ↑       ↑       ↑       ↑
    =#
    Ta, s, Tb = tsvd(gate, ((1, 3), (2, 4)); trunc=truncerr(1e-15))
    Ta, Tb = absorb_s(Ta, s, Tb)
    #= apply gate to peps

                    ↗            ↗
                 Ta ----←---- Tb
                ↗           ↗
            ↓ ↗         ↓ ↗                 ↓ ↗     ↓ ↗
        -←- A ----←---- B -←-    ===>   -←- A ==<== B -←-
            ↓           ↓                   ↓       ↓
    =#
    cp1 = _next(col, Nc)
    A, B = peps.A[row, col], peps.A[row, cp1]
    @assert !isdual(domain(A)[2])
    @tensor A[d n s w; e1 e2] := Ta[d da e1] * A[da; n e2 s w]
    @tensor B[d n e s; w1 w2] := Tb[w1 d db] * B[db; n e s w2]
    #= perform QR (The numbers show axis order of each tensor)

            ↓ ↗     ↓ ↗
        -←- A ==<== B -←-
            ↓       ↓

            2  1                                        2  1
            ↓ ↗                                         ↓ ↗
        5 ← X ← 3    1 ← Ra<= 2/3   2/3 <=Rb → 1    5 → Y ← 3
            ↓                                           ↓
            4                                           4
    =#
    X, Ra = leftorth!(A)
    Y, Rb = leftorth!(B)
    X = permute(X, ((1,), (2, 5, 3, 4)))
    Y = permute(Y, ((1,), Tuple(2:5)))
    # construct environment of Ra, Rb
    env = bondenv_ntu(row, col, X, Y, peps, alg.bondenv_alg)
    @assert [isdual(space(env, ax)) for ax in 1:4] == [0, 0, 1, 1]
    #= simple SVD truncation or environment assisted truncation (EAT) between Ra, Rb

        ← Ra =<= Rb →  =  ← Ma ← Mb →
    =#
    @tensor RaRb[-1; -2] := Ra[-1, 1, 2] * Rb[-2, 1, 2]
    Ma, Mb = absorb_s(tsvd(RaRb; trunc=alg.trscheme)[1:3]...)
    # further optimize Ma, Mb using alternating least square algorithm
    Ma, Mb, cost = als_optimize(Ma, Mb, RaRb, env, alg.als_alg)
    #= truncate A, B with Ma, Mb

            -2 -1                           -1 -2
             ↓ ↗                             ↓ ↗
        -5 ← X ← 1 ← Ma ← -3   -5 ← Mb → 1 → Y ← -3
             ↓                               ↓
            -4                               -4
    =#
    @tensor A[-1; -2 -3 -4 -5] := X[-1 -2 1 -4 -5] * Ma[1 -3]
    @tensor B[-1; -2 -3 -4 -5] := Y[-1 -2 -3 -4 1] * Mb[-5 1]
    # update peps with normalized A, B
    peps.A[row, col] = A / norm(A, Inf)
    peps.A[row, cp1] = B / norm(B, Inf)
    return nothing
end

"""
One round of update of all nearest neighbor bonds in InfinitePEPS

Reference: 
- Physical Review B 104, 094411 (2021)
- Physical Review B 106, 195105 (2022)
"""
function bondupdate_iter(gate::LocalOperator, peps::InfinitePEPS, alg::PEPSUpdate)
    @assert size(gate.lattice) == size(peps)
    Nr, Nc = size(peps)
    # TODO: make algorithm independent on the choice of dual in the network
    for (r, c) in Iterators.product(1:Nr, 1:Nc)
        @assert [isdual(space(peps.vertices[r, c], ax)) for ax in 1:5] == [0, 1, 1, 0, 0]
        @assert [isdual(space(peps.weights[1, r, c], ax)) for ax in 1:2] == [0, 1]
        @assert [isdual(space(peps.weights[2, r, c], ax)) for ax in 1:2] == [0, 1]
    end
    peps2 = deepcopy(peps)
    gate_mirrored = mirror_antidiag(gate)
    for direction in 1:2
        # mirror the y-weights to x-direction 
        # to update them using code for x-weights
        if direction == 2
            peps2 = mirror_antidiag(peps2)
        end
        for site in CartesianIndices(peps2.vertices)
            r, c = Tuple(site)
            term = get_gateterm(
                direction == 1 ? gate : gate_mirrored,
                (CartesianIndex(r, c), CartesianIndex(r, c + 1)),
            )
            ϵ = _update_bondx!(r, c, term, peps2, alg)
        end
        if direction == 2
            peps2 = mirror_antidiag(peps2)
        end
    end
    return peps2
end

"""
Perform PEPS update with nearest neighbor Hamiltonian `ham`, where the evolution
information is printed every `check_int` steps. 
"""
function bondupdate(peps::InfiniteWeightPEPS, ham::LocalOperator, alg::PEPSUpdate)
    error("Not implemented")
    time_start = time()
    Nr, Nc = size(peps)
    if bipartite
        @assert Nr == Nc == 2
    end
    # exponentiating the 2-site Hamiltonian gate
    gate = get_gate(alg.dt, ham)
    for count in 1:(alg.maxiter)
        time0 = time()
        peps = bondupdate_iter(gate, peps, alg)
        cancel = (count == alg.maxiter)
    end
    return peps
end
