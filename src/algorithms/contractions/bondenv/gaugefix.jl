"""
Replace bond environment `benv` by its positive approximant `Z† Z`
(returns the "half" environment `Z`)
```
    ┌-----------------┐     ┌---------------┐
    | ┌----┐          |     |               |
    └-|    |-- 3  4 --┘     └-- Z -- 3  4 --┘
      |benv|            =       ↓
    ┌-|    |-- 1  2 --┐     ┌-- Z†-- 1  2 --┐
    | └----┘          |     |               |
    └-----------------┘     └---------------┘
```
"""
function positive_approx(benv::BondEnv)
    # hermitize `benv` and perform eigen-decomposition
    # benv = U D U'
    D, U = eigh((benv + benv') / 2)
    # determine if `env` is (mostly) positive or negative
    sgn = sign(mean(vcat((diag(b) for (k, b) in blocks(D))...)))
    # When optimizing the truncation of a bond, 
    # its environment can always be multiplied by a number.
    # If `benv` is negative (e.g. obtained approximately from CTMRG), 
    # we can multiply it by (-1).
    (sgn == -1) && (D *= -1)
    # set (small) negative eigenvalues to 0
    for (k, b) in blocks(D)
        for i in diagind(b)
            (b[i] < 0) && (b[i] = 0.0)
        end
    end
    Z = sdiag_pow(D, 0.5) * U'
    return Z
end

"""
Use QR decomposition to fix gauge of the half bond environment `Z`.
The reduced bond tensors `a`, `b` and `Z` are arranged as
```
    ┌---------------┐
    |               |
    └---Z---a---b---┘
        |   ↓   ↓
        ↓
```
Reference: Physical Review B 92, 035142 (2015)
"""
function fixgauge_benv(
    Z::AbstractTensorMap{T,S,1,2}, 
    a::AbstractTensorMap{T,S,1,2}, 
    b::AbstractTensorMap{T,S,2,1}
) where {T<:Number,S<:ElementarySpace}
    @assert !isdual(space(Z, 1))
    @assert !isdual(space(a, 2))
    @assert !isdual(space(b, 2))
    #= QR/LQ decomposition of Z 

        3 - Z - 2   =   1 - L - 2   1 - QL - 3
            ↓                           ↓
            1                           2

                    =   1 - QR - 3  1 - R - 2
                            ↓
                            2
    =#
    L, QL = rightorth(Z, ((3,), (1, 2)))
    QR, R = leftorth(Z, ((3, 1), (2,)))
    Linv, Rinv = inv(L), inv(R)
    #= fix gauge of Z, a, b
        ┌------------------------------------┐
        |                                    |
        └---Z--Rinv)--(R--a)--(b--L)--(Linv--┘
            |              ↓    ↓
            ↓
        
        -1 - R - 1 - a - -3   -1 - b - 1 - L - -3
                     ↓             ↓        
                    -2            -2

        ┌-----------------------------------------┐
        |                                         |
        └---Z-- 1 --Rinv-- -2      -3 --Linv-- 2 -┘
            ↓
            -1
    =#
    @plansor a[-1; -2 -3] := R[-1; 1] * a[1; -2 -3]
    @plansor b[-1 -2; -3] := b[-1 -2; 1] * L[1; -3]
    @plansor Z[-1; -2 -3] := Linv[-3; 2] * Z[-1; 1 2] * Rinv[1; -2]
    (isdual(space(R, 1)) == isdual(space(R, 2))) && twist!(a, 1)
    (isdual(space(L, 1)) == isdual(space(L, 2))) && twist!(b, 3)
    return Z, a, b, (Linv, Rinv)
end

"""
When the (half) bond environment `Z` consists of two `PEPSOrth` tensors `X`, `Y` as
```
    ┌---------------┐   ┌-------------------┐
    |               | = |                   | ,
    └---Z--       --┘   └--Z0---X--    --Y--┘
        ↓                  ↓
```
apply the gauge transformation `Linv`, `Rinv` for `Z` to `X`, `Y`:
```
        -1                                     -1
         |                                      |
    -4 - X - 1 - Rinv - -2      -4 - Linv - 1 - Y - -2
         |                                      |
        -3                                     -3
```
"""
function _fixgauge_benvXY(
    X::PEPSOrth{T,S},
    Y::PEPSOrth{T,S},
    Linv::AbstractTensorMap{T,S,1,1},
    Rinv::AbstractTensorMap{T,S,1,1},
) where {T<:Number,S<:ElementarySpace}
    @plansor X[-1 -2 -3 -4] := X[-1 1 -3 -4] * Rinv[1; -2]
    @plansor Y[-1 -2 -3 -4] := Linv[-4; 1] * Y[-1 -2 -3 1]
    return X, Y
end
