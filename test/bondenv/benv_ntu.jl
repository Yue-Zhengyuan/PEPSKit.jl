using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using KrylovKit
using Random

function _benv_condition_number(benv::AbstractTensorMap)
    @assert codomain(benv) == domain(benv)
    u, s, vh, = tsvd(benv)
    return PEPSKit._condition_number(s)
end

Nr, Nc = 2, 2
Random.seed!(20)
Pspace = Vect[FermionParity](0 => 1, 1 => 1)
V2 = Vect[FermionParity](0 => 1, 1 => 1)
V3 = Vect[FermionParity](0 => 1, 1 => 2)
V4 = Vect[FermionParity](0 => 2, 1 => 2)
V5 = Vect[FermionParity](0 => 3, 1 => 2)
W1 = Vect[FermionParity](0 => 2, 1 => 3)
W2 = Vect[FermionParity](0 => 4, 1 => 1)
Pspaces = fill(Pspace, (Nr, Nc))
Nspaces = [V2 V2; V4 V4]
Espaces = [V3 V5; V5 V3]

peps = InfiniteWeightPEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
for I in CartesianIndices(peps.vertices)
    peps.vertices[I] /= norm(peps.vertices[I], Inf)
end
# The NTU bond environments are constructed exactly
# and should be positive definite
for env_alg in (NTUEnvNN(), NTUEnvNNN(), NTUEnvNNNp())
    @info "Testing $(typeof(env_alg))"
    for row in 1:Nr, col in 1:Nc
        cp1 = PEPSKit._next(col, Nc)
        A, B = peps.vertices[row, col], peps.vertices[row, cp1]
        X, a, b, Y = PEPSKit._qr_bond(A, B)
        benv = PEPSKit.bondenv_ntu(row, col, X, Y, peps, env_alg)
        # @assert [isdual(space(benv, ax)) for ax in 1:numind(benv)] == [0, 0, 1, 1]
        ab = PEPSKit._combine_ab(a, b)
        nrm1 = PEPSKit.inner_prod(benv, ab, ab)
        # benv should be Hermitian
        @test benv' ≈ benv
        # benv should be positive definite
        D, U = eigh(benv)
        @test all(all(x -> x >= 0, diag(b)) for (k, b) in blocks(D))
        @assert benv ≈ U * D * U'
        cond = _benv_condition_number(benv)
        # make condition number smaller by gauge fixing
        Z = PEPSKit.sdiag_pow(D, 0.5) * U'
        @assert benv ≈ Z' * Z
        Z2, a2, b2, (Linv, Rinv) = PEPSKit.fixgauge_benv(Z, a, b)
        benv2 = Z2' * Z2
        cond2 = _benv_condition_number(benv2)
        @test cond2 < cond
        @info "benv cond number: (gauge-fixed) $(cond2) < $(cond) (initial)"
        a2b2 = PEPSKit._combine_ab(a2, b2)
        nrm2 = PEPSKit.inner_prod(benv2, a2b2, a2b2)
        @test nrm1 ≈ nrm2
        # verify gauge transformation of X, Y
        X2, Y2 = PEPSKit._fixgauge_benvXY(X, Y, Linv, Rinv)
        benv3 = PEPSKit.bondenv_ntu(row, col, X2, Y2, peps, env_alg)
        cond3 = _benv_condition_number(benv3)
        @info norm(benv2 - benv3, Inf)
        @info "benv3 cond number: $(cond3)"
        # full contraction before and after gauge fixing
        nrm3 = PEPSKit.inner_prod(benv3, a2b2, a2b2)
        @info nrm2
        @info nrm3
    end
end
