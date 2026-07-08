using Test
using Random
using LinearAlgebra
using TensorKit
using PEPSKit
using PEPSKit: virtual_projector

"""
Cost function of LocalTruncation.
For test convenience, open virtual indices except
east/west legs are made trivial and removed.
"""
function localcompress_cost(A1, A2, B1, B2, P1, P2)
    @tensor net1[pa1 pb1 Dw1 Dw2; pa2′ pb2′ De1 De2] :=
        A1[pa1 pa; D1 Dw1] * A2[pa pa2′; D2 Dw2] * B1[pb1 pb; De1 D1] * B2[pb pb2′; De2 D2]
    @tensor net2[pa1 pb1 Dw1 Dw2; pa2′ pb2′ De1 De2] := P1[Da1 Da2; D] * P2[D; Db1 Db2] *
        A1[pa1 pa; Da1 Dw1] * A2[pa pa2′; Da2 Dw2] * B1[pb1 pb; De1 Db1] * B2[pb pb2′; De2 Db2]
    return norm(net1 - net2)
end

@testset "Fermionic twists" begin
    Vphy = Vect[FermionParity](0 => 2, 1 => 2)
    Vvir = Vect[FermionParity](0 => 2, 1 => 2)
    for _ in 1:4 # multiple trials without setting seed
        Aspace = (Vphy ⊗ Vphy' ← Vvir ⊗ Vvir ⊗ Vvir' ⊗ Vvir')
        A1 = randn(ComplexF64, Aspace)
        A2 = randn(ComplexF64, Aspace)
        for MM in [PEPSKit._get_MMdag(A1, A2), PEPSKit._get_MdagM(A1, A2)]
            @test isposdef(MM)
        end
    end
end

@testset "Cost function of LocalTruncation" begin
    Random.seed!(0)
    Vaux, Vvir, Vphy, V = ℂ^1, ℂ^4, ℂ^3, ℂ^4
    A1 = normalize(randn(Vphy ⊗ Vphy' ← Vaux ⊗ V ⊗ Vaux' ⊗ Vvir'), Inf)
    A2 = normalize(randn(Vphy ⊗ Vphy' ← Vaux ⊗ V ⊗ Vaux' ⊗ Vvir'), Inf)
    B1 = normalize(randn(Vphy ⊗ Vphy' ← Vaux ⊗ Vvir ⊗ Vaux' ⊗ V'), Inf)
    B2 = normalize(randn(Vphy ⊗ Vphy' ← Vaux ⊗ Vvir ⊗ Vaux' ⊗ V'), Inf)

    errs = map((false, true)) do layerwise_qr
        for _ in 1:5
            @time P1, P2, info = virtual_projector(A1, A2, B1, B2; trunc = notrunc(), layerwise_qr)
        end
        @test P1 * P2 ≈ TensorKit.id(domain(P2))

        P1, P2, info = virtual_projector(A1, A2, B1, B2; trunc = truncrank(8), layerwise_qr)
        # keep west virtual leg
        A1′ = removeunit(removeunit(A1, 5), 3)
        A2′ = removeunit(removeunit(A2, 5), 3)
        # keep east virtual leg
        B1′ = removeunit(removeunit(B1, 5), 3)
        B2′ = removeunit(removeunit(B2, 5), 3)
        err = info.ϵ
        @info "Truncation error = $(err)."
        @test err ≈ localcompress_cost(A1′, A2′, B1′, B2′, P1, P2)
        return err
    end
end

@testset "Virtual space matching" begin
    Vps = ComplexSpace.([2 2; 2 2])
    Vns = ComplexSpace.([2 4; 5 3])
    Ves = ComplexSpace.([3 5; 4 2])
    ρ = InfinitePEPO(randn, ComplexF64, Vps, Vns, Ves)
    for layerwise_qr in (false, true)
        alg = LocalTruncation(; trunc = truncrank(2), layerwise_qr)
        ρ2, = compress((ρ, ρ), alg)
        @test ρ2 isa InfinitePEPO
    end
end
