# using Test
# using SobolSampling
# using MLJBase
# using MLJTuning
# using Sobol
# using Random

# TODO: update tests to be compatible code in with src/main.jl

# @testset "SobolSampling.jl Tests" begin

#     # ========================================================================
#     # Test SobolSearch Construction
#     # ========================================================================

#     @testset "SobolSearch Constructor" begin
#         # Test default construction
#         tuning = SobolSearch()
#         @test tuning.skip_initial == 0
#         @test tuning.bounded == true
#         @test tuning.optimize_skip == true
#         @test tuning.deterministic == true

#         # Test custom construction
#         tuning = SobolSearch(
#             skip_initial = 15,
#             bounded = false,
#             optimize_skip = false,
#             deterministic = false
#         )
#         @test tuning.skip_initial == 15
#         @test tuning.bounded == false
#         @test tuning.optimize_skip == false
#         @test tuning.deterministic == false
#     end

#     # ========================================================================
#     # Test clean! Method
#     # ========================================================================

#     @testset "clean! Method" begin
#         # Test negative skip_initial correction
#         tuning = SobolSearch(skip_initial = -5)
#         msg = MLJTuning.clean!(tuning)
#         @test tuning.skip_initial == 0
#         @test occursin("non-negative", msg)

#         # Test skip_initial adjustment to 2^m - 1 form
#         tuning = SobolSearch(skip_initial = 10)
#         msg = MLJTuning.clean!(tuning)
#         @test tuning.skip_initial in [7, 15]  # Nearest 2^m - 1 values
#         @test occursin("2^m - 1", msg)

#         # Test valid skip_initial (already 2^m - 1)
#         tuning = SobolSearch(skip_initial = 31)  # 2^5 - 1
#         msg = MLJTuning.clean!(tuning)
#         @test tuning.skip_initial == 31
#         @test msg == ""
#     end

#     # ========================================================================
#     # Test Range Processing
#     # ========================================================================

#     @testset "Range Processing" begin
#         # Test continuous range
#         r1 = range(Float64, :x1, lower=0.0, upper=1.0)
#         ranges, bounds, cat_idx, disc_idx, scales, fields =
#             SobolSampling.process_ranges([r1])

#         @test length(ranges) == 1
#         @test bounds[1] == (0.0, 1.0)
#         @test isempty(cat_idx)
#         @test isempty(disc_idx)
#         @test scales[1] == :linear

#         # Test discrete range
#         r2 = range(Int, :x2, lower=1, upper=10)
#         ranges, bounds, cat_idx, disc_idx, scales, fields =
#             SobolSampling.process_ranges([r2])

#         @test length(ranges) == 1
#         @test bounds[1] == (1.0, 10.0)
#         @test isempty(cat_idx)
#         @test 1 in disc_idx

#         # Test categorical range
#         r3 = range(String, :x3, values=["a", "b", "c"])
#         ranges, bounds, cat_idx, disc_idx, scales, fields =
#             SobolSampling.process_ranges([r3])

#         @test length(ranges) == 1
#         @test bounds[1] == (0.0, 1.0)
#         @test 1 in cat_idx
#         @test isempty(disc_idx)

#         # Test mixed ranges
#         mixed = [r1, r2, r3]
#         ranges, bounds, cat_idx, disc_idx, scales, fields =
#             SobolSampling.process_ranges(mixed)

#         @test length(ranges) == 3
#         @test length(bounds) == 3
#         @test cat_idx == [3]
#         @test disc_idx == [2]
#         @test length(scales) == 3
#     end

#     # ========================================================================
#     # Test Sobol Point Scaling
#     # ========================================================================

#     @testset "Sobol Point Scaling" begin
#         # Create a simple state for testing
#         r1 = range(Float64, :x1, lower=0.0, upper=10.0)
#         r2 = range(Int, :x2, lower=1, upper=5)
#         r3 = range(String, :x3, values=["a", "b", "c"])

#         ranges, bounds, cat_idx, disc_idx, scales, fields =
#             SobolSampling.process_ranges([r1, r2, r3])

#         state = SobolSampling.SobolSearchState(
#             nothing, ranges, bounds, cat_idx, disc_idx, scales, fields, 0, 3
#         )

#         # Test scaling with point in [0,1]^3
#         point = [0.5, 0.5, 0.33]
#         scaled = SobolSampling.scale_sobol_point(point, state)

#         @test scaled[1] ≈ 5.0  # Continuous: 0 + 0.5 * 10
#         @test scaled[2] in 1:5  # Discrete integer
#         @test scaled[3] in ["a", "b", "c"]  # Categorical

#         # Test boundary cases
#         point_low = [0.0, 0.0, 0.0]
#         scaled_low = SobolSampling.scale_sobol_point(point_low, state)
#         @test scaled_low[1] ≈ 0.0
#         @test scaled_low[2] == 1
#         @test scaled_low[3] == "a"

#         point_high = [1.0, 1.0, 0.99]
#         scaled_high = SobolSampling.scale_sobol_point(point_high, state)
#         @test scaled_high[1] ≈ 10.0
#         @test scaled_high[2] == 5
#         @test scaled_high[3] == "c"
#     end

#     # ========================================================================
#     # Test default_n Method
#     # ========================================================================

#     @testset "default_n Method" begin
#         tuning = SobolSearch()

#         # Test with different dimensions
#         r1 = [range(Float64, Symbol("x$i"), lower=0.0, upper=1.0) for i in 1:2]
#         @test MLJTuning.default_n(tuning, r1) == 64  # 2^6 for 2D

#         r2 = [range(Float64, Symbol("x$i"), lower=0.0, upper=1.0) for i in 1:5]
#         @test MLJTuning.default_n(tuning, r2) == 256  # 2^8 for 5D

#         r3 = [range(Float64, Symbol("x$i"), lower=0.0, upper=1.0) for i in 1:15]
#         @test MLJTuning.default_n(tuning, r3) == 1024  # 2^10 for 15D

#         r4 = [range(Float64, Symbol("x$i"), lower=0.0, upper=1.0) for i in 1:25]
#         @test MLJTuning.default_n(tuning, r4) == 2048  # 2^11 for 25D
#     end

#     # ========================================================================
#     # Integration Test with Mock Model
#     # ========================================================================

#     @testset "Integration with MLJTuning" begin
#         # Create a simple mock model
#         mutable struct MockModel <: MLJBase.Deterministic
#             param1::Float64
#             param2::Int
#             param3::String
#         end

#         MockModel(; param1=0.5, param2=2, param3="a") =
#             MockModel(param1, param2, param3)

#         # Define ranges
#         r1 = range(Float64, :param1, lower=0.0, upper=1.0)
#         r2 = range(Int, :param2, lower=1, upper=5)
#         r3 = range(String, :param3, values=["a", "b", "c"])
#         ranges = [r1, r2, r3]

#         # Create tuning strategy
#         tuning = SobolSearch(skip_initial=0, bounded=false)

#         # Test setup
#         model = MockModel()
#         n = 16
#         state = MLJTuning.setup(tuning, model, ranges, n, 0)

#         @test state isa SobolSampling.SobolSearchState
#         @test state.dimension == 3
#         @test state.generated_count == 0
#         @test state.sobol_seq isa SobolSeq

#         # Test model generation
#         models, new_state = MLJTuning.models(tuning, model, [], state, 4, 0)

#         @test length(models) == 4
#         @test all(m isa MockModel for m in models)
#         @test new_state.generated_count == 4

#         # Check that generated models have valid parameters
#         for m in models
#             @test 0.0 <= m.param1 <= 1.0
#             @test m.param2 in 1:5
#             @test m.param3 in ["a", "b", "c"]
#         end

#         # Test another batch
#         models2, new_state2 = MLJTuning.models(tuning, model, [], new_state, 4, 0)
#         @test length(models2) == 4
#         @test new_state2.generated_count == 8
#     end

#     # ========================================================================
#     # Test extras and tuning_report Methods
#     # ========================================================================

#     @testset "extras and tuning_report Methods" begin
#         tuning = SobolSearch()

#         # Create a simple state
#         state = SobolSampling.SobolSearchState(
#             SobolSeq(3), [], [], [3], [2], [], [:x1, :x2, :x3], 10, 3
#         )

#         # Test extras
#         extras = MLJTuning.extras(tuning, [], state, nothing)
#         @test extras.sobol_index == 10
#         @test extras.dimension == 3
#         @test extras.using_bounds == true
#         @test extras.has_categorical == true
#         @test extras.has_discrete == true

#         # Test tuning_report
#         report = MLJTuning.tuning_report(tuning, [], state)
#         @test report.total_generated == 10
#         @test report.dimensions == 3
#         @test report.categorical_params == 1
#         @test report.discrete_params == 1
#         @test report.continuous_params == 1
#         @test report.parameter_names == [:x1, :x2, :x3]
#     end

#     # ========================================================================
#     # Test Bounded vs Unbounded Mode
#     # ========================================================================

#     @testset "Bounded vs Unbounded Mode" begin
#         # Test bounded mode (no categorical parameters)
#         r1 = range(Float64, :x1, lower=0.0, upper=10.0)
#         r2 = range(Float64, :x2, lower=1.0, upper=5.0)
#         ranges = [r1, r2]

#         tuning_bounded = SobolSearch(bounded=true)
#         model = NamedTuple()  # Mock model
#         state = MLJTuning.setup(tuning_bounded, model, ranges, 16, 0)

#         @test state.sobol_seq isa SobolSeq
#         @test length(state.sobol_seq.lb) == 2
#         @test state.sobol_seq.lb ≈ [0.0, 1.0]
#         @test state.sobol_seq.ub ≈ [10.0, 5.0]

#         # Test unbounded mode (forced)
#         tuning_unbounded = SobolSearch(bounded=false)
#         state = MLJTuning.setup(tuning_unbounded, model, ranges, 16, 0)

#         @test state.sobol_seq isa SobolSeq
#         @test isempty(state.sobol_seq.lb)  # Unit hypercube mode
#     end

#     # ========================================================================
#     # Test Skip Optimization
#     # ========================================================================

#     @testset "Skip Optimization" begin
#         r = range(Float64, :x, lower=0.0, upper=1.0)
#         model = NamedTuple()

#         # Test with optimize_skip=true
#         tuning = SobolSearch(optimize_skip=true, skip_initial=0)
#         state = MLJTuning.setup(tuning, model, [r], 100, 0)
#         # Should skip 2^6 - 1 = 63 points (largest 2^m - 1 ≤ 50)
#         @test state.generated_count == 0  # But counter starts at 0

#         # Test with optimize_skip=false
#         tuning = SobolSearch(optimize_skip=false, skip_initial=7)
#         state = MLJTuning.setup(tuning, model, [r], 100, 0)
#         @test state.generated_count == 0  # Counter still starts at 0
#     end

#     # ========================================================================
#     # Test Log Scale Support
#     # ========================================================================

#     @testset "Log Scale Support" begin
#         # Create range with log scale
#         r = range(Float64, :x, lower=0.1, upper=100.0, scale=:log)
#         ranges, bounds, cat_idx, disc_idx, scales, fields =
#             SobolSampling.process_ranges([r])

#         @test scales[1] == :log

#         state = SobolSampling.SobolSearchState(
#             nothing, ranges, bounds, cat_idx, disc_idx, scales, fields, 0, 1
#         )

#         # Test that log scaling works properly
#         point = [0.5]  # Middle of [0,1]
#         scaled = SobolSampling.scale_sobol_point(point, state)

#         # On log scale, middle point should be geometric mean
#         expected = sqrt(0.1 * 100.0)  # ≈ 3.16
#         @test scaled[1] ≈ expected rtol=0.01
#     end

#     # ========================================================================
#     # Test Edge Cases
#     # ========================================================================

#     @testset "Edge Cases" begin
#         # Test single parameter
#         r = range(Float64, :x, lower=0.0, upper=1.0)
#         tuning = SobolSearch()
#         model = NamedTuple()
#         state = MLJTuning.setup(tuning, model, [r], 10, 0)

#         @test state.dimension == 1
#         models, _ = MLJTuning.models(tuning, model, [], state, 1, 0)
#         @test length(models) == 1

#         # Test empty range (should handle gracefully)
#         state = MLJTuning.setup(tuning, model, [], 10, 0)
#         @test state.dimension == 0
#     end

# end
