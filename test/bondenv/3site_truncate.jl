using Random
using Printf
using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using PEPSKit: cost_function_als, _flip_virtuals!, _cluster_truncate!

Random.seed!(0)
maxiter = 600
check_interval = 20
trunc = truncerror(; atol = 1.0e-10) & truncrank(2)
alg = ALSTruncation(; trunc, maxiter, check_interval)

#= Index dimensions
                    Dd
                    ↓
                    b
                    ↓ ↘
                    DD′ d
                    ↓
    Dd -←-a-←-DD′-←-M-←- D
          ↘         ↓ ↘
            d       D  d²
Mimicking the situation of an iPEPO with physical dimension d, 
virtual dimension D, updated with an MPO with bond dimension D′.
=#
@testset "3-site iterative optimization ($S)" for S in [Z2Irrep, FermionParity]
    d, D, D′ = 2, 4, 2
    Dd, DD′ = D * d, D * D′
    hd, hD, hD′ = div(d, 2), div(D, 2), div(D′, 2)
    hDd, hDD = div(Dd, 2), div(DD′, 2)
    Vext = Vect[S](0 => 600, 1 => 600)
    VDd = Vect[S](0 => hDd, 1 => hDd)
    VDD = Vect[S](0 => hDD, 1 => hDD)
    VD = Vect[S](0 => hD, 1 => hD)
    Vd = Vect[S](0 => hd, 1 => hd)
    elt = Float64
    # random positive-definite environment
    Vbond = VDd ⊗ VD' ⊗ VD ⊗ VDd'
    Z = randn(elt, Vext ← Vbond)
    benv = Z' * Z
    normalize!(benv, Inf)
    # untruncated bond tensor
    Ms = [
        randn(elt, VDd ⊗ Vd ← VDD),
        randn(elt, VDD ⊗ fuse(Vd, Vd) ⊗ VD' ⊗ VD ← VDD'),
        randn(elt, VDD' ⊗ Vd ← VDd),
    ]
    normalize!.(Ms, Inf)
    # Vidal gauge truncation
    flips = [isdual(space(M, 1)) for M in Ms[2:end]]
    Ms_trunc = deepcopy(Ms)
    _flip_virtuals!(Ms_trunc, flips)
    _cluster_truncate!(Ms_trunc, fill(trunc, 2))
    _flip_virtuals!(Ms_trunc, flips)
    cost0, fid0 = cost_function_als(benv, Ms_trunc, Ms)
    @info "Fidelity of truncated Vidal gauge = $fid0.\n"
    # 3-site iterative optimization
    Ms_trunc, wts, info = PEPSKit.se3site_truncate(Ms, benv, alg)
    @info "Improved fidelity = $(info.fid)."
    @test info.fid ≈ cost_function_als(benv, Ms_trunc, Ms)[2]
    @test info.fid > fid0
end
