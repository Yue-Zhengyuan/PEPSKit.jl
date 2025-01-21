"""
    FullEnvTruncation

Algorithm struct for the full environment truncation (FET).
"""
@kwdef struct FullEnvTruncation
    tol::Float64 = 1e-10
    maxiter::Int = 100
    trscheme::TensorKit.TruncationScheme
    verbose::Bool = false
    check_int::Int = 10
end

"""
Given the bond environment `env`, calculate the inner product
between two states specified by the bond matrices `b1`, `b2`
```
                ┌──←──┐   ┌──←──┐
                │     │   │     │
                │   ┌─┴───┴─┐   │
    ⟨b1|b2⟩ =   b1† │  env  │   b2
                │   └─┬───┬─┘   │
                │     │   │     │
                └──←──┘   └──←──┘
```
"""
function inner_prod(
    env::AbstractTensorMap{S,2,2}, b1::AbstractTensor{S,2}, b2::AbstractTensor{S,2}
) where {S<:ElementarySpace}
    # dual check
    @assert [isdual(space(env, ax)) for ax in 1:4] == [0, 0, 1, 1]
    val = @tensor conj(b1[1 2]) * env[1 2; 3 4] * b2[3 4]
    return val
end

"""
Given the bond environment `env`, calculate the fidelity
between two states specified by the bond matrices `b1`, `b2`
```
    F(b1, b2) = (⟨b1|b2⟩ ⟨b2|b1⟩) / (⟨b1|b1⟩ ⟨b2|b2⟩)
```
"""
function fidelity(
    env::AbstractTensorMap{S,2,2}, b1::AbstractTensor{S,2}, b2::AbstractTensor{S,2}
) where {S<:ElementarySpace}
    return abs2(inner_prod(env, b1, b2)) /
           real(inner_prod(env, b1, b1) * inner_prod(env, b2, b2))
end

"""
Given a fixed state `|b0⟩` with bond matrix `b0`, 
find the state `|b⟩` with truncated bond matrix `b = u s v†`
that maximizes the fidelity (not normalized by `⟨b0|b0⟩`)
```
    F(b) = ⟨b|b0⟩⟨b0|b⟩ / ⟨b|b⟩

                ┌──←──┐   ┌──←──┐   ┌──←──┐   ┌──←──┐
                v     │   │     │   │     │   │     v†
                ↓   ┌─┴───┴─┐   │   │   ┌─┴───┴─┐   ↑
                s   │  env  │   b0  b0† │  env  │   s
                ↑   └─┬───┬─┘   │   │   └─┬───┬─┘   ↓
                u†    │   │     │   │     │   │     u
                └──←──┘   └──←──┘   └──←──┘   └──←──┘
            = ──────────────────────────────────────────
                        ┌──←──┐   ┌──←──┐
                        v     │   │     v†
                        ↓   ┌─┴───┴─┐   ↑
                        s   │  env  │   s
                        ↑   └─┬───┬─┘   ↓
                        u†    │   │     u
                        └──←──┘   └──←──┘
```
- The bond environment `env` is positive definite 
    (Hermitian with positive (at least non-negative) eigenvalues). 
- The singular value spectrum of `s` is truncated to desired bond dimension.

The algorithm iteratively optimizes the vectors `l`, `r`
```
                      ┌─┐                     ┌─┐
          ┌─┐         │ ↑         ┌─┐         │ ↑
        ←─┘ │       ←─┘ s       ←─┘ │       ←─┘ v†
            l   =       ↓   ,       r   =       ↑
        ←─┐ │       ←─┐ u       ←─┐ │       ←─┐ s 
          └─┘         │ ↓         └─┘         │ ↓
                      └─┘                     └─┘
```

## Optimization of `r`

Define the vector `p` and the positive map `B` as
```
                ┌───┐           ┌─←─┐   ┌───┐
                │   │           │   │   │   │  
                │   └─←         │  ┌┴───┴┐  └─←
                p†          =  b0† │ env │ 
                │   ┌─←         │  └┬───┬┘  ┌─←
                │   │           │   │   │   u
                └───┘           └─←─┘   └───┘

          ┌───┐   ┌───┐         ┌───┐   ┌───┐
          │   │   │   │         │   │   │   │
        ←─┘  ┌┴───┴┐  └─←     ←─┘  ┌┴───┴┐  └─←
             │  B  │        =      │ env │
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
    ∂F/∂(r̄_a) * (r† B r)²
    = p (p† r) (r† B r) - |p† r|² (B r) = 0
```
Note that `B` is positive (consequently `B† = B`). 
Then the solution for the vector `r` is
```
    r = B⁻¹ p
```
We can verify that (using `B† = B`)
```
    ∂F/∂(r̄_a) * (r† B r)²
    = p (p† B⁻¹ p) (p† B⁻¹ B B⁻¹ p) - |p† B⁻¹ p|² (B B⁻¹ p) 
    = 0
```
Then the bond matrix `u s v†` is updated by truncated SVD:
```
    ← u ← r →    ==>    ← u ← s → v† →
```

## Optimization of `l`

The process is entirely similar. 
Define the vector `p` and the positive map `B` as
```
                ┌───┐           ┌─←─┐   ┌───┐
                │   │           │   │   │   v†
                │   └─←         │  ┌┴───┴┐  └─←
                p†          =  b0† │ env │ 
                │   ┌─←         │  └┬───┬┘  ┌─←
                │   │           │   │   │   │
                └───┘           └─←─┘   └───┘

          ┌───┐   ┌───┐         ┌───┐   ┌───┐
          │   │   │   │         v   │   │   v†
        ←─┘  ┌┴───┴┐  └─←     ←─┘  ┌┴───┴┐  └─←
             │  B  │        =      │ env │
        ←─┐  └┬───┬┘  ┌─←     ←─┐  └┬───┬┘  ┌─←
          │   │   │   │         │   │   │   │
          └───┘   └───┘         └───┘   └───┘
```
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
    ← l → v† →   ==>    ← u ← s → v† →
```

## Returns

- When `flip_s` is `false`, return `← u ← s ← v† →`
- When `flip_s` is `true`, return `← u ← s → v† →`

Reference: Physical Review B 98, 085155 (2018)
"""
function fullenv_truncate(
    env::AbstractTensorMap{S,2,2},
    b0::AbstractTensor{S,2},
    alg::FullEnvTruncation;
    flip_s::Bool=false,
) where {S<:ElementarySpace}
    # sanity check
    @assert space(b0, 1) == space(b0, 2)
    @assert [isdual(space(env, ax)) for ax in 1:4] == [0, 0, 1, 1]
    @assert [isdual(space(b0, ax)) for ax in 1:2] == [0, 0]
    # initilize `u, s, v†` using (almost) un-truncated bond matrix
    u, s, vh = flip_svd(tsvd(b0, ((1,), (2,)); trunc=truncerr(1e-15))...)
    s0 = deepcopy(s)
    diff, fid = NaN, 0.0
    if alg.verbose
        @info "Initial bond space = $(space(s, 1))\n"
    end
    for iter in 1:(alg.maxiter)
        time0 = time()
        # update `r`
        @tensor r[-1 -2] := s[-1 1] * vh[1 -2]
        @tensor p[-1 -2] := conj(u[1 -1]) * env[1 -2; 3 4] * b0[3 4]
        @tensor B[-1 -2; -3 -4] := conj(u[1 -1]) * env[1 -2; 3 -4] * u[3 -3]
        @assert [isdual(space(B, ax)) for ax in 1:4] == [0, 0, 1, 1]
        r, info_r = linsolve(x -> B * x, p, r, 0, 1)
        @tensor b1[-1; -2] := u[-1 1] * r[1 -2]
        u, s, vh = flip_svd(tsvd(b1; trunc=alg.trscheme)...)
        # update `l`
        @tensor l[-1 -2] := u[-1 1] * s[1 -2]
        @tensor p[-1 -2] := conj(vh[-2 2]) * env[-1 2; 3 4] * b0[3 4]
        @tensor B[-1 -2; -3 -4] := conj(vh[-2 2]) * env[-1 2; -3 4] * vh[-4 4]
        @assert [isdual(space(B, ax)) for ax in 1:4] == [0, 0, 1, 1]
        l, info_l = linsolve(x -> B * x, p, l, 0, 1)
        @tensor b1[-1; -2] := l[-1 1] * vh[1 -2]
        u, s, vh = flip_svd(tsvd(b1; trunc=alg.trscheme)...)
        # determine convergence
        diff = (space(s) == space(s0)) ? _singular_value_distance((s, s0)) : NaN
        fid = fidelity(env, b0, permute(b1, (1, 2)))
        s0 = deepcopy(s)
        time1 = time()
        message = @sprintf(
            "Iter %4d,  diff = %12.6e,  fidelity = %8.6f,  time = %.3f s\n",
            iter,
            diff,
            fid,
            time1 - time0
        )
        if iter == alg.maxiter
            @warn "Maximal iteration $(alg.maxiter) reached"
            @warn message
            @warn "Bond space = $(space(s, 1))\n"
        end
        if alg.verbose && (iter == 1 || iter % alg.check_int == 0 || diff < alg.tol)
            @info message
            @info "Bond space = $(space(s, 1))\n"
        end
        if diff < alg.tol
            break
        end
    end
    @tensor tmp[-1; -2] := u[-1 1] * s[1 2] * vh[2 -2]
    if !flip_s
        # change `← u ← s → v† →` back to `← u ← s ← v† →`
        Vtrunc = space(s, 1)
        flipper = isomorphism(flip(Vtrunc), Vtrunc)
        # ← u ←(← s → f ←) ← (← f† → v† →)→
        @tensor s[-1; -2] := s[-1 1] * flipper[1 -2]
        # TODO: figure out the reason behind
        twist!(s, 2)
        @tensor vh[-1; -2] := flipper'[-1 1] * vh[1 -2]
        @assert tmp ≈ u * s * vh
    end
    return u, s, vh, (diff, fid)
end
