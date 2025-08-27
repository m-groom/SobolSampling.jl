using Aqua

@testset "Aqua.jl" begin
    Aqua.test_all(
        SobolSampling;
        ambiguities=true,
        unbound_args=true,
        undefined_exports=true,
        project_extras=true,
        deps_compat=true,
        persistent_tasks=false,
    )
    Aqua.test_persistent_tasks(SobolSampling; tmax=120)
end
