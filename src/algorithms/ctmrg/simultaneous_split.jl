"""
    SimultaneousSplitCTMRG(; kwargs...)

Split-CTMRG with all directions and unit-cell positions updated from the same environment.
Uses the projector constructions and keyword arguments of [`SplitCTMRG`](@ref).
Corners are expanded in both directions and truncated with the composed green/teal projectors of their adjoining edges.
The corner contractions reuse joined edges; the updated edges retain separate ket and bra factors.
"""
struct SimultaneousSplitCTMRG{P <: SplitProjector, I <: TruncationStrategy} <: AbstractSplitCTMRG
    tol::Float64
    maxiter::Int
    miniter::Int
    verbosity::Int
    projector_alg::P
    interlayer_trunc::I
end
SimultaneousSplitCTMRG(; kwargs...) = CTMRGAlgorithm(; alg = :SimultaneousSplitCTMRG, kwargs...)::SimultaneousSplitCTMRG
CTMRG_SYMBOLS[:SimultaneousSplitCTMRG] = SimultaneousSplitCTMRG

"""Compose the outer projectors at the two endpoints of each updated west edge."""
function _split_outer_projectors(projectors)
    (; green, teal) = projectors
    left = map(eachindex(green[1])) do r
        G, T = green[1][r], teal[1][r]
        return @tensor P[χold Dk Db; χnew] := G[χold Dk; χg] * T[χg Db; χnew]
    end
    right = map(eachindex(green[2])) do r
        nextrow = _next(r, length(green[2]))
        G, T = green[2][nextrow], teal[2][nextrow]
        return @tensor P[χnew; χold Dk Db] := T[χnew; χg Db] * G[χg; χold Dk]
    end
    return left, right
end

"""Restore the spatial coordinates of a boundary computed in the left-move orientation."""
function _restore_split_boundary(data, dir::Int)
    dir == NORTH && return permutedims(data)
    dir == EAST && return reverse(data; dims = 2)
    dir == SOUTH && return rotl90(data)
    return data
end

"""Compute one complete boundary from a frozen environment using the existing split left-move routines."""
function _simultaneous_split_direction(network, env, alg::SimultaneousSplitCTMRG, dir::Int)
    oriented_network, oriented_env = _orient_split_move(network, env, dir)
    columns = map(axes(oriented_network, 2)) do col
        projectors, info = Base.inferencebarrier(sequential_projectors)(
            col, oriented_network, oriented_env, alg.projector_alg, alg.interlayer_trunc,
        )
        updated = Base.inferencebarrier(renormalize_sequentially)(col, projectors, oriented_network, oriented_env)
        left, right = _split_outer_projectors(projectors)
        return updated.edges_ket[WEST, :, col], updated.edges_bra[WEST, :, col], left, right, info.truncation_error
    end
    # Pack each field into a flat tensor array before indexing it in a pullback.
    K = hcat(map(first, columns)...)
    B = hcat(map(x -> x[2], columns)...)
    L = hcat(map(x -> x[3], columns)...)
    R = hcat(map(x -> x[4], columns)...)
    if dir == NORTH || dir == EAST
        K, B = _reflect_split_tensor.(K), _reflect_split_tensor.(B)
        L, R = copy.(adjoint.(R)), copy.(adjoint.(L))
    end
    return (
        _restore_split_boundary(K, dir), _restore_split_boundary(B, dir),
        _restore_split_boundary(L, dir), _restore_split_boundary(R, dir),
        maximum(last, columns),
    )
end

function ctmrg_iteration(network::InfiniteSquareNetwork, env::SplitCTMRGEnv, alg::SimultaneousSplitCTMRG)
    moves = map(dir -> _simultaneous_split_direction(network, env, alg, dir), 1:4)
    K = permutedims(cat(map(first, moves)...; dims = 3), (3, 1, 2))
    B = permutedims(cat(map(x -> x[2], moves)...; dims = 3), (3, 1, 2))
    left = cat(map(x -> x[3], moves)...; dims = 3)
    right = cat(map(x -> x[4], moves)...; dims = 3)
    joined = CTMRGEnv(env)
    corner_updates = (
        renormalize_northwest_corner, renormalize_northeast_corner,
        renormalize_southeast_corner, renormalize_southwest_corner,
    )
    corners = map(eachcoordinate(env, 1:4)) do (dir, r, c)
        Q = EnlargedCorner(network, joined, (dir, r, c))
        C = corner_updates[dir](Q, left[r, c, dir], right[r, c, _prev(dir, 4)])
        return _normalize_split_tensor(C, env.corners[dir, r, c])
    end
    truncation_error = maximum(last, moves)
    return SplitCTMRGEnv(corners, K, B, env.ket_first), (; contraction_metrics = (; truncation_error))
end
