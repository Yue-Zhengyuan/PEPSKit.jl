using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using KrylovKit
using Random

Vphy = Vect[FermionParity](0 => 10, 1 => 10)
V = Vect[FermionParity](0 => 2, 1 => 2)
Vs = (V, V')
#= 
    ┌---------------------------┐
    |                           |
    └---Z-- 1 --a-- 2 --b-- 3 --┘
        ↓       ↓       ↓
        -1      -2      -3
=#
for V1 in Vs, V2 in Vs, V3 in Vs
    Z = randn(ComplexF64, Vphy ← V1 ⊗ V3')
    a = randn(ComplexF64, V1 ← Vphy' ⊗ V2)
    b = randn(ComplexF64, V2 ⊗ Vphy ← V3)
    @tensor half[:] := Z[-1; 1 3] * a[1; -2 2] * b[2 -3; 3]
    Z, a, b, = PEPSKit.fixgauge_benv(Z, a, b)
    @tensor half2[:] := Z[-1; 1 3] * a[1; -2 2] * b[2 -3; 3]
    @test half ≈ half2
end
