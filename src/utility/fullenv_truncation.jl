"""
    FullEnvTruncation

Algorithm struct for the full environment truncation (FET).
"""
@kwdef struct FullEnvTruncation
    trscheme::TensorKit.TruncationScheme
    maxiter::Int = 50
    tol::Float64 = 1e-15
    check_int::Int = 0
end

"""
Given the bond environment `benv`, calculate the inner product
between two states specified by the bond matrices `b1`, `b2`
```
                ┌─────┐   ┌─────┐
                │     │   │     │
                │   ┌─┴───┴─┐   │
    ⟨b1|b2⟩ =   b1† │ benv  │   b2
                │   └─┬───┬─┘   │
                │     │   │     │
                └─────┘   └─────┘
```
"""
function inner_prod(
    benv::AbstractTensorMap{T,S,2,2}, b1::AbstractTensor{T,S,2}, b2::AbstractTensor{T,S,2}
) where {T<:Number,S<:ElementarySpace}
    val = @tensor conj(b1[1 2]) * benv[1 2; 3 4] * b2[3 4]
    return val
end

"""
Given the bond environment `benv`, calculate the fidelity
between two states specified by the bond matrices `b1`, `b2`
```
    F(b1, b2) = (⟨b1|b2⟩ ⟨b2|b1⟩) / (⟨b1|b1⟩ ⟨b2|b2⟩)
```
"""
function fidelity(
    benv::AbstractTensorMap{T,S,2,2}, b1::AbstractTensor{T,S,2}, b2::AbstractTensor{T,S,2}
) where {T<:Number,S<:ElementarySpace}
    return abs2(inner_prod(benv, b1, b2)) /
           real(inner_prod(benv, b1, b1) * inner_prod(benv, b2, b2))
end

"""
Apply a twist to domain or codomain indices that correspond to dual spaces
"""
function _linearmap_twist!(t::AbstractTensorMap)
    for ax in 1:numout(t)
        isdual(codomain(t, ax)) && twist!(t, ax)
    end
    for ax in 1:numin(t)
        isdual(domain(t, ax)) && twist!(t, numout(t) + ax)
    end
    return nothing
end

function _fet_message(
    iter::Int, fid::Float64, Δfid::Float64, Δwt::Float64, time_elapsed::Float64
)
    return @sprintf("%5d: fid = %.8e, Δfid = %.8e, ", iter, fid, Δfid) *
           @sprintf("|Δs| = %.6e, time = %.2e s", Δwt, time_elapsed)
end

"""
    fullenv_truncate(benv::AbstractTensorMap{T,S,2,2}, b0::AbstractTensor{T,S,2}, alg::FullEnvTruncation) where {T<:Number,S<:ElementarySpace}

The full environment truncation algorithm
(Physical Review B 98, 085155 (2018)). 
Given a fixed state `|b0⟩` with bond matrix `b0`
and the corresponding positive-definite bond environment `benv`, 
find the state `|b⟩` with truncated bond matrix `b = u s v†`
that maximizes the fidelity (not normalized by `⟨b0|b0⟩`)
```
    F(b) = ⟨b|b0⟩⟨b0|b⟩ / ⟨b|b⟩

                ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
                v     │   │     │   │     │   │     v†
                ↑   ┌─┴───┴─┐   │   │   ┌─┴───┴─┐   ↓
                s   │ benv  │   b0  b0† │ benv  │   s
                ↑   └─┬───┬─┘   │   │   └─┬───┬─┘   ↓
                u†    │   │     │   │     │   │     u
                └─────┘   └─────┘   └─────┘   └─────┘
            = ──────────────────────────────────────────
                        ┌─────┐   ┌─────┐
                        v     │   │     v†
                        ↑   ┌─┴───┴─┐   ↓
                        s   │ benv  │   s
                        ↑   └─┬───┬─┘   ↓
                        u†    │   │     u
                        └─────┘   └─────┘
```
The singular value spectrum `s` is truncated to desired dimension, 
and normalized such that the maximum is 1.

The algorithm iteratively optimizes the vectors `l`, `r`
```
                      ┌─┐                     ┌─┐
          ┌─┐         │ ↓         ┌─┐         │ │
        →─┘ │       ──┘ s       ──┘ │       ──┘ v†
            l   =       ↓   ,       r   =       ↓
        ──┐ │       ──┐ u       ←─┐ │       ←─┐ s 
          └─┘         │ │         └─┘         │ ↓
                      └─┘                     └─┘
```

## Optimization of `r`

Define the vector `p` and the positive map `B` as
```
                ┌───┐           ┌───┐   ┌───┐
                │   │           │   │   │   │  
                │   └──         │  ┌┴───┴┐  └──
                p†          =  b0† │benv │ 
                │   ┌─←         │  └┬───┬┘  ┌─←
                │   │           │   │   │   u
                └───┘           └───┘   └───┘

          ┌───┐   ┌───┐         ┌───┐   ┌───┐
          │   │   │   │         │   │   │   │
        ──┘  ┌┴───┴┐  └──     ──┘  ┌┴───┴┐  └──
             │  B  │        =      │benv │
        ←─┐  └┬───┬┘  ┌─←     ←─┐  └┬───┬┘  ┌─←
          │   │   │   │         u†  │   │   u
          └───┘   └───┘         └───┘   └───┘
```
Then (each index corresponds to a pair of fused indices)
```
    F(r,r†) = |p† r|² / (r† B r)
            = (r† p) (p† r) / (r† B r)
```
which is maximized when
```
    ∂F/∂r̄ * (r† B r)²
    = p (p† r) (r† B r) - |p† r|² (B r) = 0
```
Note that `B` is positive (consequently `B† = B`). 
Then the solution for the vector `r` is
```
    r = B⁻¹ p
```
We can verify that (using `B† = B`)
```
    ∂F/∂r̄ * (r† B r)²
    = p (p† B⁻¹ p) (p† B⁻¹ B B⁻¹ p) - |p† B⁻¹ p|² (B B⁻¹ p) 
    = 0
```
Then the bond matrix `u s v†` is updated by truncated SVD:
```
    ← u ← r →    ==>    ← u ← s ← v† →
```

## Optimization of `l`

The process is entirely similar. 
Define the vector `p` and the positive map `B` as
```
                ┌───┐           ┌───┐   ┌───┐
                │   │           │   │   │   v†
                │   └o→         │  ┌┴───┴┐  └o→
                p†          =  b0† │benv │ 
                │   ┌─←         │  └┬───┬┘  ┌──
                │   │           │   │   │   │
                └───┘           └───┘   └───┘

          ┌───┐   ┌───┐         ┌───┐   ┌───┐
          │   │   │   │         v   │   │   v†
        →o┘  ┌┴───┴┐  └o→     →o┘  ┌┴───┴┐  └o→
             │  B  │        =      │benv │
        ──┐  └┬───┬┘  ┌──     ──┐  └┬───┬┘  ┌──
          │   │   │   │         │   │   │   │
          └───┘   └───┘         └───┘   └───┘
```
Here `o` is the parity tensor (twist) necessary for fermions. 
Then (each index corresponds to a pair of fused indices)
```
    F(l,l†) = |p† l|² / (l† B l)
```
which is maximized when
```
    l = B⁻¹ p
```
Then the bond matrix `u s v†` is updated by SVD:
```
    ← l ← v† →   ==>    ← u ← s ← v† →
```

## Returns

The SVD result of the new bond matrix `u`, `s`, `vh`.
"""
function fullenv_truncate(
    benv::AbstractTensorMap{T,S,2,2}, b0::AbstractTensor{T,S,2}, alg::FullEnvTruncation
) where {T<:Number,S<:ElementarySpace}
    verbose = (alg.check_int > 0)
    time00 = time()
    # initialize u, s, vh with truncated SVD
    u, s, vh = tsvd(b0, ((1,), (2,)); trunc=alg.trscheme)
    # normalize `s` (bond matrices can always be normalized)
    s /= norm(s, Inf)
    s0 = deepcopy(s)
    Δfid, Δs, fid, fid0 = NaN, NaN, 0.0, 0.0
    for iter in 1:(alg.maxiter)
        time0 = time()
        # update `← r -  =  ← s ← v† -`
        @tensor r[-1 -2] := s[-1 1] * vh[1 -2]
        @tensor p[-1 -2] := conj(u[1 -1]) * benv[1 -2; 3 4] * b0[3 4]
        @tensor B[-1 -2; -3 -4] := conj(u[1 -1]) * benv[1 -2; 3 -4] * u[3 -3]
        _linearmap_twist!(p)
        _linearmap_twist!(B)
        r, info_r = linsolve(x -> B * x, p, r, 0, 1)
        @tensor b1[-1; -2] := u[-1 1] * r[1 -2]
        u, s, vh = tsvd(b1; trunc=alg.trscheme)
        s /= norm(s, Inf)
        # update `- l ←  =  - u ← s ←`
        @tensor l[-1 -2] := u[-1 1] * s[1 -2]
        @tensor p[-1 -2] := conj(vh[-2 2]) * benv[-1 2; 3 4] * b0[3 4]
        @tensor B[-1 -2; -3 -4] := conj(vh[-2 2]) * benv[-1 2; -3 4] * vh[-4 4]
        _linearmap_twist!(p)
        _linearmap_twist!(B)
        l, info_l = linsolve(x -> B * x, p, l, 0, 1)
        @tensor b1[-1; -2] := l[-1 1] * vh[1 -2]
        u, s, vh = tsvd(b1; trunc=alg.trscheme)
        s /= norm(s, Inf)
        # determine convergence
        fid = fidelity(benv, b0, permute(b1, (1, 2)))
        Δs = (space(s) == space(s0)) ? _singular_value_distance((s, s0)) : NaN
        Δfid = fid - fid0
        s0 = deepcopy(s)
        fid0 = fid
        # @assert diff_fid >= -1e-14 "Fidelity is decreasing by $diff_fid."
        time1 = time()
        converge = (Δfid < alg.tol)
        cancel = (iter == alg.maxiter)
        showinfo = verbose && (converge || cancel || iter == 1 || iter % alg.check_int == 0)
        if showinfo
            message = _fet_message(
                iter, fid, Δfid, Δs, time1 - ((cancel || converge) ? time00 : time0)
            )
            if converge
                @info "FET conv" * message
            elseif cancel
                @warn "FET cancel" * message
            else
                @info "FET iter" * message
            end
        end
        converge && break
    end
    return u, s, vh, (; fid, Δfid, Δs)
end
