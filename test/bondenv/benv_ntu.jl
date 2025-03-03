using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using KrylovKit
using Random

function _benv_condition_number(benv::AbstractTensorMap)
    @assert codomain(benv) == domain(benv)
    s = tsvd(benv)[2]
    return PEPSKit._condition_number(s)
end

Nr, Nc = 2, 2
Random.seed!(20)
Pspace = Vect[FermionParity](0 => 1, 1 => 1)
V2 = Vect[FermionParity](0 => 1, 1 => 1)
V3 = Vect[FermionParity](0 => 1, 1 => 2)
V4 = Vect[FermionParity](0 => 2, 1 => 2)
V5 = Vect[FermionParity](0 => 3, 1 => 2)
Pspaces = fill(Pspace, (Nr, Nc))
Nspaces = [V2 V2; V4 V4]
Espaces = [V3 V5; V5 V3]

peps = InfiniteWeightPEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
for I in CartesianIndices(peps.vertices)
    peps.vertices[I] /= norm(peps.vertices[I], Inf)
end
for env_alg in (NTUEnvNN(), NTUEnvNNN(), NTUEnvNNNp())
    @info "Testing $(typeof(env_alg))"
    for row in 1:Nr, col in 1:Nc
        cp1 = PEPSKit._next(col, Nc)
        A, B = peps.vertices[row, col], peps.vertices[row, cp1]
        X, a, b, Y = PEPSKit._qr_bond(A, B)
        @tensor ab[DX DY; da db] := a[DX da D] * b[D db DY]
        benv = PEPSKit.bondenv_ntu(row, col, X, Y, peps, env_alg)
        # NTU bond environments are constructed exactly
        # and should be positive definite
        @test benv' ≈ benv
        @assert [isdual(space(benv, ax)) for ax in 1:numind(benv)] == [0, 0, 1, 1]
        nrm1 = PEPSKit.inner_prod(benv, ab, ab)
        @test nrm1 ≈ real(nrm1)
        D, U = eigh(benv)
        @test all(all(x -> x >= 0, diag(b)) for (k, b) in blocks(D))
        @assert benv ≈ U * D * U'
        # gauge fixing
        Z = PEPSKit.sdiag_pow(D, 0.5) * U'
        @assert benv ≈ Z' * Z
        Z2, a2, b2, (Linv, Rinv) = PEPSKit.fixgauge_benv(Z, a, b)
        benv2 = Z2' * Z2
        # gauge fixing should reduce condition number
        cond = _benv_condition_number(benv)
        cond2 = _benv_condition_number(benv2)
        @test cond2 <= cond
        @info "benv cond number: (gauge-fixed) $(cond2) ≤ $(cond) (initial)"
        # verify gauge transformation of X, Y
        @tensor a2b2[DX DY; da db] := a2[DX da D] * b2[D db DY]
        nrm2 = PEPSKit.inner_prod(benv2, a2b2, a2b2)
        X2, Y2 = PEPSKit._fixgauge_benvXY(X, Y, Linv, Rinv)
        benv3 = PEPSKit.bondenv_ntu(row, col, X2, Y2, peps, env_alg)
        benv3 *= norm(benv2, Inf)
        nrm3 = PEPSKit.inner_prod(benv3, a2b2, a2b2)
        @test benv2 ≈ benv3
        @test nrm1 ≈ nrm2 ≈ nrm3
    end
end
