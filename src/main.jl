"""
    SobolSequence(; skip = :auto, random_shift=false, rng=Random.GLOBAL_RNG)

Instantiate a grid-based hyperparameter tuning strategy that generates a
Sobol low-discrepancy plan in `[0,1)^d` and maps it to the user-specified
hyperparameter `range`.

- `skip = :auto` uses Sobol.jl's recommended skipping of `2^m - 1 ≤ n` points
  when the total plan size `n` is known (improves uniformity for finite `n`).
  Pass a nonnegative integer to skip exactly that many points instead.
- `random_shift = true` applies a Cranley–Patterson uniform shift mod 1, using
  `rng`, yielding a simple scramble-like variant (deterministic if `rng` is
  seeded).
- `rng` can be an `AbstractRNG` or an integer seed (converted to a
  `Random.MersenneTwister`).

### Supported ranges

`range` can be:
- A single one-dimensional MLJ `ParamRange`
- A vector of such ranges

Both `NumericRange`s and `NominalRange`s are supported. Numeric ranges honour
their `scale` (e.g. `:log`), and integer-typed numeric ranges are rounded after
inverse scaling. Unbounded numeric ranges are bounded following the same
heuristics used in `LatinHypercube`, centered using `origin`/`unit`.

See also: Sobol.jl (`SobolSeq`, `skip`), and `TunedModel` docs for usage.
"""

# --------------------------
# Strategy type
# --------------------------

mutable struct SobolSequence <: MLJTuning.TuningStrategy
    skip::Union{Int,Symbol}
    random_shift::Bool
    rng::Random.AbstractRNG
end

function SobolSequence(; skip=:auto, random_shift=false, rng=Random.GLOBAL_RNG)
    _rng = rng isa Integer ? Random.MersenneTwister(rng) : rng
    seq = SobolSequence(skip, random_shift, _rng)
    message = MLJTuning.clean!(seq)
    isempty(message) || @warn message

    return seq
end

function MLJTuning.clean!(tuning::SobolSequence)
    err = ""
    if tuning.skip isa Symbol && tuning.skip !== :auto
        err *= "`skip` must be :auto or a nonnegative integer. Resetting to :auto."
        tuning.skip = :auto
    elseif tuning.skip isa Integer && tuning.skip < 0
        err *= "`skip` must be nonnegative. Resetting to 0"
        tuning.skip = 0
    end
    return err
end

# --------------------------
# Range helpers
# --------------------------

# Produce scaled-space bounds for each range and remember types.
# For NumericRange, work in "scale space" then inverse_transform back.
# For NominalRange, just record the number of categories.
function _scaled_bounds_and_kinds(ranges::Vector)
    d = length(ranges)
    bounds = Vector{Tuple{Float64,Float64}}(undef, d)  # only for numeric dims
    kinds  = Vector{Symbol}(undef, d)                  # :numeric or :nominal
    card   = Vector{Int}(undef, d)                     # K for nominal

    for (i, r) in enumerate(ranges)
        if r isa NumericRange
            sc = MLJBase.scale(r.scale)
            if !(r.scale isa Symbol)
                throw(ArgumentError("Callable scales are not supported."))
            end
            # For log scale, ensure positivity of any finite bounds or origin used
            if r.scale === :log
                if isfinite(r.lower) && !(r.lower > 0)
                    throw(ArgumentError("For :log scale, `lower` must be > 0."))
                end
                if isfinite(r.upper) && !(r.upper > 0)
                    throw(ArgumentError("For :log scale, `upper` must be > 0."))
                end
                if !isfinite(r.lower) && !isfinite(r.upper) && !(r.origin > 0)
                    throw(ArgumentError("For :log scale with unbounded range, `origin` must be > 0."))
                end
            end
            # Finite-bounds heuristic as in LatinHypercube:
            lower_fin = isfinite(r.lower)
            upper_fin = isfinite(r.upper)
            a, b = if lower_fin && upper_fin
                (transform(MLJBase.Scale, sc, r.lower),
                 transform(MLJBase.Scale, sc, r.upper))
            elseif !lower_fin && upper_fin
                u = transform(MLJBase.Scale, sc, r.upper)
                (u - 2r.unit, u)
            elseif lower_fin && !upper_fin
                l = transform(MLJBase.Scale, sc, r.lower)
                (l, l + 2r.unit)
            else
                o = transform(MLJBase.Scale, sc, r.origin)
                (o - r.unit, o + r.unit)
            end
            if !(isfinite(a) && isfinite(b))
                throw(ArgumentError("Scaled bounds are non-finite for $(r.field) with scale=$(r.scale). Check lower/upper/origin/unit."))
            end
            bounds[i] = (Float64(a), Float64(b))
            kinds[i]  = :numeric
            card[i]   = 0
        else
            # NominalRange
            kinds[i] = :nominal
            card[i]  = length(r.values)
            bounds[i] = (0.0, 1.0)  # placeholder, unused for nominal
        end
    end
    return bounds, kinds, card
end

# Map a unit value u in [0,1) to the numeric range r, given scaled-space bounds (a,b).
# Linearly map in scale space, then inverse_transform and round if integer-typed.
@inline function _denorm_numeric(r::NumericRange{T}, u::Float64, a::Float64, b::Float64) where {T<:Real}
    sc = MLJBase.scale(r.scale)
    x_scaled = a + u*(b - a)
    x = inverse_transform(MLJBase.Scale, sc, x_scaled)
    if !isfinite(x)
        throw(DomainError(x, "Non-finite value after inverse scaling for $(r.field) with scale=$(r.scale). Check bounds/origin/unit."))
    end
    if T <: Integer
        # Clamp to finite bounds if available before rounding
        if isfinite(r.lower) && isfinite(r.upper)
            x = clamp(x, float(r.lower), float(r.upper))
        end
        x = round(T, x)
    end
    return x
end

# Map a unit value u in [0,1) to a nominal index in 1:K (uniform categorical).
@inline _nominal_index(u::Float64, K::Int) = min(K, floor(Int, u*K) + 1)

# --------------------------
# Plan generation
# --------------------------

# Generate an n×d matrix of unit Sobol points, optionally with a random shift.
function _unit_sobol_matrix(d::Int, n::Int, skip_option, random_shift::Bool, rng::AbstractRNG)
    s = SobolSeq(d)
    if skip_option === :auto
        skip(s, n)  # recommended skip (2^m - 1)
    elseif skip_option isa Integer
        @assert skip_option >= 0 "`skip` must be nonnegative."
        skip(s, skip_option, exact=true)
    else
        throw(ArgumentError("`skip` must be :auto or a nonnegative integer."))
    end
    # optional Cranley–Patterson shift:
    δ = random_shift ? rand(rng, d) : zeros(Float64, d)
    U = Matrix{Float64}(undef, n, d)
    @inbounds for i in 1:n
        u = next!(s)                # Vector{Float64} length d in [0,1)
        @simd for j in 1:d
            U[i, j] = (u[j] + δ[j]) % 1.0
        end
    end
    return U
end

# Scale a unit plan U to the provided ranges.
function _rescale_plan(U::AbstractMatrix{<:Real}, ranges::Vector,
                       bounds::Vector{Tuple{Float64,Float64}},
                       kinds::Vector{Symbol}, card::Vector{Int})
    n, d = size(U)
    # Return as a vector of d tuples, each is the column of values per dim
    cols = Vector{Vector}(undef, d)
    for (j, r) in enumerate(ranges)
        if kinds[j] === :numeric
            a, b = bounds[j]
            vj = Vector{Any}(undef, n)
            @inbounds for i in 1:n
                vj[i] = _denorm_numeric(r, U[i,j], a, b)
            end
            cols[j] = vj
        else
            K = card[j]
            vj = Vector{Any}(undef, n)
            @inbounds for i in 1:n
                idx = _nominal_index(U[i,j], K)
                vj[i] = r.values[idx]
            end
            cols[j] = vj
        end
    end
    return cols  # length d, each length n
end

# --------------------------
# Strategy API
# --------------------------

function MLJTuning.setup(tuning::SobolSequence, model, range, n::Int, verbosity::Int)
    ranges = range isa AbstractVector ? collect(range) : [range]
    d = length(ranges)

    if d == 0
        # No parameters to tune - just create n copies of the model
        models = [deepcopy(model) for _ in 1:n]
        state = (models=models, fields=Symbol[], parameter_scales=Symbol[])
        return state
    end

    bounds, kinds, card = _scaled_bounds_and_kinds(ranges)
    U = _unit_sobol_matrix(d, n, tuning.skip, tuning.random_shift, tuning.rng)
    cols = _rescale_plan(U, ranges, bounds, kinds, card)

    fields = map(r -> r.field, ranges)
    parameter_scales = scale.(ranges)

    models = _make_models_from_plan(model, fields, cols)  # size n

    state = (models=models, fields=fields, parameter_scales=parameter_scales)
    return state
end

# Hand out all remaining models
function MLJTuning.models(tuning::SobolSequence, model, history, state, n_remaining::Int, verbosity::Int)
    return state.models[MLJTuning._length(history)+1:end], state
end

# Optional: enable nice plotting in `report`.
MLJTuning.tuning_report(tuning::SobolSequence, history, state) =
    (plotting = MLJTuning.plotting_report(state.fields, state.parameter_scales, history),)

# Provide a conservative default when `n` is not specified.
MLJTuning.default_n(::SobolSequence, range) = 128

# Build clones by writing field values from the column-wise plan.
function _make_models_from_plan(prototype::Model, fields, cols::Vector{<:Vector})
    N = length(first(cols))
    map(1:N) do i
        clone = deepcopy(prototype)
        for k in eachindex(fields)
            recursive_setproperty!(clone, fields[k], cols[k][i])
        end
        clone
    end
end
