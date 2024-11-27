using Test
using Printf
using Random
using PEPSKit
using TensorKit
import Statistics: mean
include("utility/measure_heis.jl")
import .MeasureHeis: measure_heis

# benchmark data is from Phys. Rev. B 94, 035133 (2016)

# random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
Dcut, χenv = 4, 16
N1, N2 = 2, 2
Random.seed!(0)
peps = InfiniteWeightPEPS(rand, Float64, ℂ^2, ℂ^Dcut; unitcell=(N1, N2))
# normalize vertex tensors
for ind in CartesianIndices(peps.vertices)
    peps.vertices[ind] /= norm(peps.vertices[ind], Inf)
end
# Heisenberg model Hamiltonian
# (already only includes nearest neighbor terms)
ham = heisenberg_XYZ(InfiniteSquare(N1, N2); Jx=1.0, Jy=1.0, Jz=1.0)
# convert to real tensors
ham = LocalOperator(ham.lattice, Tuple(ind => real(op) for (ind, op) in ham.terms)...)

# simple update
dts = [1e-2, 1e-3, 4e-4, 1e-4]
tols = [1e-6, 1e-8, 1e-8, 1e-8]
maxiter = 10000
for (n, (dt, tol)) in enumerate(zip(dts, tols))
    Dcut2 = (n == 1 ? Dcut + 1 : Dcut)
    trscheme = truncerr(1e-10) & truncdim(Dcut2)
    alg = SimpleUpdate(dt, tol, maxiter, trscheme)
    result = simpleupdate(peps, ham, alg; bipartite=false)
    global peps = result[1]
end
# absort weight into site tensors
peps = InfinitePEPS(peps)
# CTMRG
envs = CTMRGEnv(rand, Float64, peps, ℂ^χenv)
trscheme = truncerr(1e-9) & truncdim(χenv)
ctm_alg = CTMRG(; tol=1e-10, verbosity=2, trscheme=trscheme, ctmrgscheme=:sequential)
envs = leading_boundary(envs, peps, ctm_alg)
# measure physical quantities
meas = measure_heis(peps, ham, envs)
display(meas)
@info @sprintf("Energy = %.8f\n", meas["e_site"])
@info @sprintf("Staggered magnetization = %.8f\n", mean(meas["mag_norm"]))
@test isapprox(meas["e_site"], -0.6675; atol=1e-3)
@test isapprox(mean(meas["mag_norm"]), 0.3767; atol=1e-3)

# continue with full update
dts = [2e-2, 1e-2, 5e-3]
trscheme_peps = truncerr(1e-10) & truncdim(Dcut)
trscheme_envs = truncerr(1e-9) & truncdim(χenv)
trscheme_envs_final = truncerr(1e-10) & truncdim(χenv)
lrmove_alg = CTMRG(;
    verbosity=0, maxiter=1, trscheme=trscheme_envs, ctmrgscheme=:sequential
)
reconv_alg = CTMRG(;
    tol=1e-6, maxiter=10, verbosity=2, trscheme=trscheme_envs, ctmrgscheme=:sequential
)
ctm_alg = CTMRG(;
    tol=1e-10,
    maxiter=50,
    verbosity=2,
    trscheme=trscheme_envs_final,
    ctmrgscheme=:sequential,
)
for dt in dts
    fu_alg = FullUpdate(;
        dt=dt,
        maxiter=1000,
        trscheme=trscheme_peps,
        lrmove_alg=lrmove_alg,
        reconv_alg=reconv_alg,
    )
    result = fullupdate(peps, envs, ham, fu_alg, ctm_alg)
    global peps = result[1]
    global envs = result[2]
end
# measure physical quantities
meas = measure_heis(peps, ham, envs)
display(meas)
@info @sprintf("Energy = %.8f\n", meas["e_site"])
@info @sprintf("Staggered magnetization = %.8f\n", mean(meas["mag_norm"]))
@test isapprox(meas["e_site"], -0.66875; atol=1e-4)
@test isapprox(mean(meas["mag_norm"]), 0.3510; atol=1e-3)
