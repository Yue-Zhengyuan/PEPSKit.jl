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

Nr, Nc = 2, 3
# create random PEPS
Pspace = Vect[FermionParity](0 => 1, 1 => 1)
Nspace = Vect[FermionParity](0 => 2, 1 => 2)
peps = InfinitePEPS(randn, ComplexF64, Pspace, Nspace; unitcell=(Nr, Nc))
for I in CartesianIndices(peps.A)
    peps.A[I] /= norm(peps.A[I], Inf)
end
# calculate CTMRG environment
Envspace = Vect[FermionParity](0 => 3, 1 => 3)
ctm_alg = SequentialCTMRG(; tol=1e-10, verbosity=2, trscheme=FixedSpaceTruncation())
env, = leading_boundary(CTMRGEnv(rand, ComplexF64, peps, Envspace), peps, ctm_alg)
for row in 1:Nr, col in 1:Nc
    cp1 = PEPSKit._next(1, Nc)
    A, B = peps.A[row, col], peps.A[row, cp1]
    X, a, b, Y = PEPSKit._qr_bond(A, B)
    # verify that gauge fixing can reduce condition number
    benv = PEPSKit.bondenv_fu(row, col, X, Y, env)
    Z = PEPSKit.positive_approx(benv)
    cond0 = _benv_condition_number(Z' * Z)
    Z, a, b = PEPSKit.fixgauge_benv(Z, a, b)
    cond1 = _benv_condition_number(Z' * Z)
    @test cond1 < cond0
    @info "benv cond number: (gauge-fixed) $(cond1) < $(cond0) (initial)"
end
