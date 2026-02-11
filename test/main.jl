using Test
using SobolSampling
using MLJBase
using MLJTuning
using Sobol
using Random

@testset "SobolSampling.jl Tests" begin

    # ========================================================================
    # Test SobolSequence Construction
    # ========================================================================

    @testset "SobolSequence Constructor" begin
        # Test default construction
        tuning = SobolSequence()
        @test tuning.skip == :auto
        @test tuning.random_shift == false
        @test tuning.rng isa Random.AbstractRNG

        # Test custom construction with all parameters
        rng = Random.MersenneTwister(123)
        tuning = SobolSequence(skip=15, random_shift=true, rng=rng)
        @test tuning.skip == 15
        @test tuning.random_shift == true
        @test tuning.rng === rng

        # Test construction with integer seed
        tuning = SobolSequence(rng=42)
        @test tuning.rng isa Random.MersenneTwister
    end

    # ========================================================================
    # Test clean! Method
    # ========================================================================

    @testset "clean! Method" begin
        # Test negative skip correction
        tuning = @test_logs (:warn, r"`skip` must be nonnegative") SobolSequence(skip=-5)
        @test tuning.skip == 0  # Should be corrected during construction

        # Test invalid symbol for skip
        tuning = SobolSequence()
        tuning.skip = :invalid
        msg = MLJTuning.clean!(tuning)
        @test tuning.skip == :auto
        @test occursin("must be :auto or a nonnegative integer", msg)

        # Test valid skip (non-negative integer)
        tuning = SobolSequence(skip=31)
        @test tuning.skip == 31

        # Test valid :auto skip
        tuning = SobolSequence(skip=:auto)
        msg = MLJTuning.clean!(tuning)
        @test tuning.skip == :auto
        @test msg == ""
    end

    # ========================================================================
    # Test Internal Helper Functions
    # ========================================================================

    @testset "Range Processing Helpers" begin
        # Test _scaled_bounds_and_kinds for continuous range
        r1 = range(Float64, :x1, lower=0.0, upper=1.0)
        bounds, kinds, card = SobolSampling._scaled_bounds_and_kinds([r1])

        @test length(bounds) == 1
        @test bounds[1] == (0.0, 1.0)
        @test kinds[1] == :numeric
        @test card[1] == 0

        # Test integer range
        r2 = range(Int, :x2, lower=1, upper=10)
        bounds, kinds, card = SobolSampling._scaled_bounds_and_kinds([r2])

        @test length(bounds) == 1
        @test bounds[1] == (1.0, 10.0)
        @test kinds[1] == :numeric
        @test card[1] == 0

        # Test categorical range
        r3 = range(String, :x3, values=["a", "b", "c"])
        bounds, kinds, card = SobolSampling._scaled_bounds_and_kinds([r3])

        @test length(bounds) == 1
        @test bounds[1] == (0.0, 1.0)  # Placeholder for nominal
        @test kinds[1] == :nominal
        @test card[1] == 3

        # Test mixed ranges
        mixed = [r1, r2, r3]
        bounds, kinds, card = SobolSampling._scaled_bounds_and_kinds(mixed)

        @test length(bounds) == 3
        @test kinds == [:numeric, :numeric, :nominal]
        @test card == [0, 0, 3]
    end

    @testset "Unit Sobol Matrix Generation" begin
        # Test basic generation
        U = SobolSampling._unit_sobol_matrix(2, 10, :auto, false, Random.GLOBAL_RNG)
        @test size(U) == (10, 2)
        @test all(0 <= u <= 1 for u in U)

        # Test with random shift
        rng = Random.MersenneTwister(123)
        U_shifted = SobolSampling._unit_sobol_matrix(2, 10, :auto, true, rng)
        @test size(U_shifted) == (10, 2)
        @test all(0 <= u < 1 for u in U_shifted)

        # Test with explicit skip
        U_skip = SobolSampling._unit_sobol_matrix(2, 10, 7, false, Random.GLOBAL_RNG)
        @test size(U_skip) == (10, 2)
    end

    @testset "Numeric Range Denormalization" begin
        # Test linear scale
        r = range(Float64, :x, lower=0.0, upper=10.0)
        val = SobolSampling._denorm_numeric(r, 0.5, 0.0, 10.0)
        @test val ≈ 5.0

        # Test integer range
        r_int = range(Int, :x, lower=1, upper=10)
        val_int = SobolSampling._denorm_numeric(r_int, 0.5, 1.0, 10.0)
        @test val_int isa Int
        @test 1 <= val_int <= 10

        # Test log scale
        r_log = range(Float64, :x, lower=0.1, upper=100.0, scale=:log)
        sc = MLJBase.scale(:log)
        a = transform(MLJBase.Scale, sc, 0.1)
        b = transform(MLJBase.Scale, sc, 100.0)
        val_log = SobolSampling._denorm_numeric(r_log, 0.5, a, b)
        # On log scale, middle point should be geometric mean
        @test val_log ≈ sqrt(0.1 * 100.0) rtol=0.01
    end

    @testset "Nominal Index Mapping" begin
        # Test categorical mapping
        @test SobolSampling._nominal_index(0.0, 3) == 1
        @test SobolSampling._nominal_index(0.33, 3) == 1  # floor(0.33*3) + 1 = 1
        @test SobolSampling._nominal_index(0.66, 3) == 2  # floor(0.66*3) + 1 = 2
        @test SobolSampling._nominal_index(0.99, 3) == 3  # floor(0.99*3) + 1 = 3
    end

    # ========================================================================
    # Test default_n Method
    # ========================================================================

    @testset "default_n Method" begin
        tuning = SobolSequence()

        # Test default value
        r = range(Float64, :x, lower=0.0, upper=1.0)
        @test MLJTuning.default_n(tuning, r) == 8
        @test MLJTuning.default_n(tuning, [r, r]) == 8^2
    end

    # ========================================================================
    # Integration Test with Mock Model
    # ========================================================================

    @testset "Integration with MLJTuning" begin
        # Create a simple mock model
        mutable struct MockModel <: MLJBase.Deterministic
            param1::Float64
            param2::Int
            param3::String
        end

        MockModel(; param1=0.5, param2=2, param3="a") = MockModel(param1, param2, param3)

        # Define ranges
        r1 = range(Float64, :param1, lower=0.0, upper=1.0)
        r2 = range(Int, :param2, lower=1, upper=5)
        r3 = range(String, :param3, values=["a", "b", "c"])
        ranges = [r1, r2, r3]

        # Create tuning strategy
        tuning = SobolSequence(skip=0, random_shift=false)

        # Test setup
        model = MockModel()
        n = 16
        state = MLJTuning.setup(tuning, model, ranges, n, 0)

        # State should contain models, fields, and parameter_scales
        @test haskey(state, :models)
        @test haskey(state, :fields)
        @test haskey(state, :parameter_scales)
        @test length(state.models) == n
        @test state.fields == [:param1, :param2, :param3]

        # Test model generation (should return all remaining models)
        history = []  # Empty history
        models, new_state = MLJTuning.models(tuning, model, history, state, 4, 0)

        @test length(models) == n  # Should return all models
        @test all(m isa MockModel for m in models)

        # Check that generated models have valid parameters
        for m in models
            @test 0.0 <= m.param1 <= 1.0
            @test m.param2 in 1:5
            @test m.param3 in ["a", "b", "c"]
        end

        # Test with partial history (simulating progress)
        history = [(model=models[1], measure=[0.1])]
        remaining_models, _ = MLJTuning.models(tuning, model, history, state, 4, 0)
        @test length(remaining_models) == n - 1
    end

    # ========================================================================
    # Test tuning_report Method
    # ========================================================================

    @testset "tuning_report Method" begin
        tuning = SobolSequence()

        # Create a mock state with models and fields
        state = (
            models=[], fields=[:x1, :x2, :x3], parameter_scales=[:linear, :log, :linear]
        )

        # Test tuning_report
        history = []
        report = MLJTuning.tuning_report(tuning, history, state)
        @test haskey(report, :plotting)
        @test report.plotting isa NamedTuple
    end

    # ========================================================================
    # Test with Different Skip Options
    # ========================================================================

    @testset "Skip Options" begin
        # Note: This test requires a proper MLJ model, not a simple struct
        # Using MockModel from the integration test
        mutable struct TestModel <: MLJBase.Deterministic
            x::Float64
        end
        TestModel(; x=0.5) = TestModel(x)

        r = range(Float64, :x, lower=0.0, upper=1.0)
        model = TestModel()

        # Test with skip=:auto
        tuning_auto = SobolSequence(skip=:auto)
        state_auto = MLJTuning.setup(tuning_auto, model, [r], 100, 0)
        @test length(state_auto.models) == 100

        # Test with explicit skip
        tuning_skip = SobolSequence(skip=7)
        state_skip = MLJTuning.setup(tuning_skip, model, [r], 100, 0)
        @test length(state_skip.models) == 100
    end

    # ========================================================================
    # Test Random Shift
    # ========================================================================

    @testset "Random Shift" begin
        # Create a simple mock model
        mutable struct TestModel2 <: MLJBase.Deterministic
            x::Float64
        end
        TestModel2(; x=0.5) = TestModel2(x)

        r = range(Float64, :x, lower=0.0, upper=1.0)
        model = TestModel2()

        # Test without random shift
        tuning_no_shift = SobolSequence(random_shift=false, rng=123)
        state1 = MLJTuning.setup(tuning_no_shift, model, [r], 10, 0)
        state2 = MLJTuning.setup(tuning_no_shift, model, [r], 10, 0)
        # Both should produce models
        @test length(state1.models) == 10
        @test length(state2.models) == 10

        # Test with random shift
        rng1 = Random.MersenneTwister(123)
        rng2 = Random.MersenneTwister(123)
        tuning_shift1 = SobolSequence(random_shift=true, rng=rng1)
        tuning_shift2 = SobolSequence(random_shift=true, rng=rng2)
        state_shift1 = MLJTuning.setup(tuning_shift1, model, [r], 10, 0)
        state_shift2 = MLJTuning.setup(tuning_shift2, model, [r], 10, 0)
        # With same seed, should produce same shifted results
        @test length(state_shift1.models) == length(state_shift2.models)
    end

    # ========================================================================
    # Test Log Scale Support
    # ========================================================================

    @testset "Log Scale Support" begin
        # Create range with log scale
        r = range(Float64, :x, lower=0.1, upper=100.0, scale=:log)
        bounds, kinds, card = SobolSampling._scaled_bounds_and_kinds([r])

        @test kinds[1] == :numeric
        # Bounds should be in log space
        sc = MLJBase.scale(:log)
        expected_a = transform(MLJBase.Scale, sc, 0.1)
        expected_b = transform(MLJBase.Scale, sc, 100.0)
        @test bounds[1][1] ≈ expected_a
        @test bounds[1][2] ≈ expected_b

        # Test error for invalid log scale range
        @test_throws ArgumentError begin
            r_invalid = range(Float64, :x, lower=-1.0, upper=100.0, scale=:log)
            SobolSampling._scaled_bounds_and_kinds([r_invalid])
        end
    end

    # ========================================================================
    # Test Unbounded Ranges
    # ========================================================================

    @testset "Unbounded Ranges" begin
        # Note: Unbounded ranges follow heuristics from LatinHypercube
        # Bounds are computed in scale space (after transform)

        # Test upper unbounded (lower finite, upper infinite)
        r_upper_inf = range(
            Float64, :x, lower=0.0, upper=Inf, origin=1.0, unit=1.0, scale=:linear
        )
        bounds, kinds, card = SobolSampling._scaled_bounds_and_kinds([r_upper_inf])
        @test isfinite(bounds[1][1]) && isfinite(bounds[1][2])
        # When lower is finite and upper is infinite: (transform(lower), transform(lower) + 2*unit)
        # Since scale is :linear, transform(0.0) = 0.0
        @test bounds[1] == (0.0, 2.0)

        # Test lower unbounded (lower infinite, upper finite)
        r_lower_inf = range(
            Float64, :x, lower=(-Inf), upper=10.0, origin=5.0, unit=2.0, scale=:linear
        )
        bounds, kinds, card = SobolSampling._scaled_bounds_and_kinds([r_lower_inf])
        @test isfinite(bounds[1][1]) && isfinite(bounds[1][2])
        # When lower is infinite and upper is finite: (transform(upper) - 2*unit, transform(upper))
        # Since scale is :linear, transform(10.0) = 10.0
        @test bounds[1] == (6.0, 10.0)

        # Test fully unbounded (both lower and upper infinite)
        r_both_inf = range(
            Float64, :x, lower=(-Inf), upper=Inf, origin=0.0, unit=5.0, scale=:linear
        )
        bounds, kinds, card = SobolSampling._scaled_bounds_and_kinds([r_both_inf])
        @test isfinite(bounds[1][1]) && isfinite(bounds[1][2])
        # When both are infinite: (transform(origin) - unit, transform(origin) + unit)
        # Since scale is :linear, transform(0.0) = 0.0
        @test bounds[1] == (-5.0, 5.0)
    end

    # ========================================================================
    # Test Edge Cases
    # ========================================================================

    @testset "Edge Cases" begin
        # Test single parameter
        mutable struct SimpleModel <: MLJBase.Deterministic
            x::Float64
        end
        SimpleModel(; x=0.5) = SimpleModel(x)

        r = range(Float64, :x, lower=0.0, upper=1.0)
        tuning = SobolSequence()
        model = SimpleModel()
        state = MLJTuning.setup(tuning, model, r, 10, 0)  # Single range, not vector

        @test length(state.models) == 10
        @test state.fields == [:x]

        models, _ = MLJTuning.models(tuning, model, [], state, 1, 0)
        @test length(models) == 10  # Returns all remaining

        # Test empty range (should handle gracefully)
        # Using a simple model with no parameters to tune
        mutable struct NoParamsModel <: MLJBase.Deterministic
            dummy::Int  # Need at least one field to avoid issues
        end
        NoParamsModel(; kwargs...) = NoParamsModel(1)

        model_no_params = NoParamsModel()
        state_empty = MLJTuning.setup(tuning, model_no_params, [], 10, 0)
        @test length(state_empty.models) == 10
        @test isempty(state_empty.fields)
    end

    # ========================================================================
    # Test Error Handling
    # ========================================================================

    @testset "Error Handling" begin
        # Test log scale with zero lower bound
        r_log_zero = range(Float64, :x, lower=0.0, upper=100.0, scale=:log)
        @test_throws ArgumentError SobolSampling._scaled_bounds_and_kinds([r_log_zero])
    end
end
