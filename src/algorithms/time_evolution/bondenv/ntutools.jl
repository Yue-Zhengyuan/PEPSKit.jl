"""
Extract tensors in an infinite PEPS at positions specified in `neighbors` relative to `(row, col)`
"""
function collect_neighbors(
    peps::InfinitePEPS, row::Int, col::Int, neighbors::Vector{Tuple{Int,Int}}
)
    Nr, Nc = size(peps)
    return Dict(
        nb => peps.A[mod1(row + nb[1], Nr), mod1(col + nb[2], Nc)] for nb in neighbors
    )
end

"""
    cal_envboundary(free_axs::Vector{Int}, ket::T, bra::T) where {T<:Union{PEPSTensor,PEPSOrth}}
    cal_envboundary(free_axs::Vector{Int}, ket::T, bra::T, axts::Vector{H}) where {T<:Union{PEPSTensor,PEPSOrth},H<:Union{Nothing,Hair}}

Contract the physical axes (for PEPSTensor) and the virtual axes of `ket` with `bra` to obtain the tensor on the boundary of the bond environment. Virtual axes specified by `free_axs` (in ascending order) are not contracted. 

# Examples (when `ket`, `bra` are PEPSTensor with a physical axis)

- Left "hair" tensor (`free_ax = 3`)
```
             ╱|
    |-----bra----- 1
    |    ╱ |  |
    |   |  |  |
    |   |  | ╱
    |---|-ket----- 2
        |╱
```

- Upper-left corner tensor (`free_ax = [3, 4]`)
```
             ╱|
    |-----bra----- 1
    |    ╱ |  |
    |   3  |  |
    |      | ╱
    |-----ket----- 2
         ╱
        4
```

- Left edge tensor (`free_ax = [2, 3, 4]`)
```
               1
             ╱
    |-----bra----- 3
    |    ╱ |
    |   5  |   2
    |      | ╱
    |-----ket----- 4
          ╱
        6
```
"""
function cal_envboundary(
    free_axs::Vector{Int}, ket::T, bra::T
) where {T<:Union{PEPSTensor,PEPSOrth}}
    if T <: PEPSTensor
        @assert all(2 <= ax <= 5 for ax in free_axs)
    else
        @assert all(1 <= ax <= 4 for ax in free_axs)
    end
    @assert issorted(free_axs)
    axs = 1:(T <: PEPSTensor ? 5 : 4)
    codomain_axes = Tuple(ax for ax in axs if !(ax in free_axs))
    domain_axes = Tuple(free_axs)
    perm = (codomain_axes, domain_axes)
    t = adjoint(permute(bra, perm)) * permute(ket, perm)
    n = length(free_axs)
    return permute(t, Tuple(Iterators.flatten(zip(1:n, (n + 1):(2n)))))
end
function cal_envboundary(
    free_axs::Vector{Int}, ket::T, bra::T, axts::Vector{H}
) where {T<:Union{PEPSTensor,PEPSOrth},H<:Union{Nothing,Hair}}
    @assert length(axts) == 4 - length(free_axs)
    ket2 = deepcopy(ket)
    axs = (T <: PEPSTensor) ? (2:5) : (1:4)
    for (axt, ax) in zip(axts, Tuple(ax for ax in axs if !(ax in free_axs)))
        if axt === nothing
            continue
        end
        @assert space(axt, 1) == space(axt, 2)'
        ket_indices = collect(-1:-1:((T <: PEPSTensor) ? -5 : -4))
        ket_indices[ax] = 1
        # apply `axt` to virtual indices of `ket` to be contracted
        ket2 = ncon([axt, ket2], [[-ax, 1], ket_indices])
    end
    if T <: PEPSTensor
        ket2 = permute(ket2, ((1,), Tuple(2:5)))
    end
    return cal_envboundary(free_axs, ket2, bra)
end

#= Free axes of different boundary tensors
(t/b/l/r mean top/bottom/left/right)
(C/E/H mean corner/edge/hair)

                                H_t
                                |
                                4

                C_tl - 3   5 - E_t - 3   5 - C_tr
                |               |               |
                4               4               4

                2               2  1            2
                |               | /             |
    H_l - 3     E_l - 3    5 - ket - 3    5 - E_r   5 - H_r
                |               |               |
                4               4               4

                2               2               2
                |               |               |
                C_bl - 3   5 - E_b - 3   5 - C_br

                                2
                                |
                                H_b
=#

hair_t(ket::PEPSTensor) = cal_envboundary([4], ket, ket)
hair_r(ket::PEPSTensor) = cal_envboundary([5], ket, ket)
hair_b(ket::PEPSTensor) = cal_envboundary([2], ket, ket)
hair_l(ket::PEPSTensor) = cal_envboundary([3], ket, ket)

hair_t(ket::PEPSOrth) = cal_envboundary([3], ket, ket)
hair_r(ket::PEPSOrth) = cal_envboundary([4], ket, ket)
hair_b(ket::PEPSOrth) = cal_envboundary([1], ket, ket)
hair_l(ket::PEPSOrth) = cal_envboundary([2], ket, ket)

cor_tl(ket::PEPSTensor) = cal_envboundary([3, 4], ket, ket)
cor_tr(ket::PEPSTensor) = cal_envboundary([4, 5], ket, ket)
cor_br(ket::PEPSTensor) = cal_envboundary([2, 5], ket, ket)
cor_bl(ket::PEPSTensor) = cal_envboundary([2, 3], ket, ket)

cor_tl(ket::PEPSOrth) = cal_envboundary([2, 3], ket, ket)
cor_tr(ket::PEPSOrth) = cal_envboundary([3, 4], ket, ket)
cor_br(ket::PEPSOrth) = cal_envboundary([1, 4], ket, ket)
cor_bl(ket::PEPSOrth) = cal_envboundary([1, 2], ket, ket)

edge_t(ket::PEPSTensor) = cal_envboundary([3, 4, 5], ket, ket)
edge_t(ket::PEPSTensor, ht) = cal_envboundary([3, 4, 5], ket, ket, [ht])
edge_r(ket::PEPSTensor) = cal_envboundary([2, 4, 5], ket, ket)
edge_r(ket::PEPSTensor, hr) = cal_envboundary([2, 4, 5], ket, ket, [hr])
edge_b(ket::PEPSTensor) = cal_envboundary([2, 3, 5], ket, ket)
edge_b(ket::PEPSTensor, hb) = cal_envboundary([2, 3, 5], ket, ket, [hb])
edge_l(ket::PEPSTensor) = cal_envboundary([2, 3, 4], ket, ket)
edge_l(ket::PEPSTensor, hl) = cal_envboundary([2, 3, 4], ket, ket, [hl])

edge_t(ket::PEPSOrth) = cal_envboundary([2, 3, 4], ket, ket)
edge_t(ket::PEPSOrth, ht) = cal_envboundary([2, 3, 4], ket, ket, [ht])
edge_r(ket::PEPSOrth) = cal_envboundary([1, 3, 4], ket, ket)
edge_r(ket::PEPSOrth, hr) = cal_envboundary([1, 3, 4], ket, ket, [hr])
edge_b(ket::PEPSOrth) = cal_envboundary([1, 2, 4], ket, ket)
edge_b(ket::PEPSOrth, hb) = cal_envboundary([1, 2, 4], ket, ket, [hb])
edge_l(ket::PEPSOrth) = cal_envboundary([1, 2, 3], ket, ket)
edge_l(ket::PEPSOrth, hl) = cal_envboundary([1, 2, 3], ket, ket, [hl])

"""
Construct the top-left corner
```
    ctl══ D1 ══ et ══ -5/-6
    ║           ║
    D2          D3
    ║           ║
    el ══ D4 ══ X ═══ -7/-8
    ║           ║
    -1/-2       -3/-4
```
"""
function enlarge_corner_tl(
    ctl::AbstractTensor{S,4},
    et::AbstractTensor{S,6},
    el::AbstractTensor{S,6},
    ket::PEPSOrth{S},
    bra::PEPSOrth{S}=ket,
) where {S<:ElementarySpace}
    return @tensor ctl2[:] :=
        ctl[D11 D10 D21 D20] *
        et[-5 -6 D31 D30 D11 D10] *
        el[D21 D20 D41 D40 -1 -2] *
        conj(bra[D31 -7 -3 D41]) *
        ket[D30 -8 -4 D40]
end

"""
Construct the bottom-right corner
```
              -1/-2       -3/-4
                ║           ║
    -5/-6 ═════ Y ══ D1 ═══ er
                ║           ║
                D2          D3
                ║           ║     
    -7/-8 ═════ eb ═ D4 ══ cbr
```
"""
function enlarge_corner_br(
    cbr::AbstractTensor{S,4},
    eb::AbstractTensor{S,6},
    er::AbstractTensor{S,6},
    ket::PEPSOrth{S},
    bra::PEPSOrth{S}=ket,
) where {S<:ElementarySpace}
    return @tensor cbr2[:] :=
        cbr[D31 D30 D41 D40] *
        eb[D21 D20 D41 D40 -7 -8] *
        er[-3 -4 D31 D30 D11 D10] *
        conj(bra[-1 D11 D21 -5]) *
        ket[-2 D10 D20 -6]
end
