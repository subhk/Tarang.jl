"""
Test suite for tools/config.jl

Tests:
1. Config creation and defaults
2. ConfigSection access
3. Type-safe getters (getboolean, getint, getfloat, getstring, getlist)
4. Default values and missing-key errors
5. Config set/get, add_section!, reset
6. TOML load/save round-trip
7. Validation
8. Environment variable overrides
"""

using Test

@testset "Configuration Module" begin
    using Tarang

    # ------------------------------------------------------------------
    # ConfigSection basics
    # ------------------------------------------------------------------
    @testset "ConfigSection construction and access" begin
        data = Dict{String,Any}("a" => 1, "b" => "hello", "flag" => true)
        sec = Tarang.ConfigSection(data, "test")

        @test sec["a"] == 1
        @test sec["b"] == "hello"
        @test haskey(sec, "flag")
        @test !haskey(sec, "missing")
        @test length(sec) == 3
        @test !isempty(sec)
        @test Set(keys(sec)) == Set(["a", "b", "flag"])

        # get with default
        @test get(sec, "a") == 1
        @test get(sec, "missing", 99) == 99

        # show
        s = sprint(show, sec)
        @test occursin("ConfigSection", s)
        @test occursin("test", s)
    end

    # ------------------------------------------------------------------
    # getboolean
    # ------------------------------------------------------------------
    @testset "getboolean" begin
        sec = Tarang.ConfigSection(Dict{String,Any}(
            "yes_bool" => true,
            "no_bool" => false,
            "yes_str" => "true",
            "no_str" => "false",
            "one_str" => "1",
            "zero_str" => "0",
            "on_str" => "on",
            "off_str" => "off",
            "num_one" => 1,
            "num_zero" => 0,
            "nan_val" => NaN,
            "bad_str" => "maybe",
        ), "bools")

        @test Tarang.getboolean(sec, "yes_bool") == true
        @test Tarang.getboolean(sec, "no_bool") == false
        @test Tarang.getboolean(sec, "yes_str") == true
        @test Tarang.getboolean(sec, "no_str") == false
        @test Tarang.getboolean(sec, "one_str") == true
        @test Tarang.getboolean(sec, "zero_str") == false
        @test Tarang.getboolean(sec, "on_str") == true
        @test Tarang.getboolean(sec, "off_str") == false
        @test Tarang.getboolean(sec, "num_one") == true
        @test Tarang.getboolean(sec, "num_zero") == false

        # NaN cannot be converted
        @test_throws ArgumentError Tarang.getboolean(sec, "nan_val")
        # Unparseable string
        @test_throws ArgumentError Tarang.getboolean(sec, "bad_str")
        # Missing key without default
        @test_throws KeyError Tarang.getboolean(sec, "nope")
        # Missing key with default
        @test Tarang.getboolean(sec, "nope"; default=true) == true
    end

    # ------------------------------------------------------------------
    # getint
    # ------------------------------------------------------------------
    @testset "getint" begin
        sec = Tarang.ConfigSection(Dict{String,Any}(
            "int_val" => 42,
            "float_int" => 5.0,
            "float_bad" => 5.5,
            "str_val" => "  100  ",
            "str_bad" => "abc",
            "inf_val" => Inf,
        ), "ints")

        @test Tarang.getint(sec, "int_val") == 42
        @test Tarang.getint(sec, "float_int") == 5
        @test Tarang.getint(sec, "str_val") == 100

        @test_throws ArgumentError Tarang.getint(sec, "float_bad")
        @test_throws ArgumentError Tarang.getint(sec, "str_bad")
        @test_throws ArgumentError Tarang.getint(sec, "inf_val")
        @test_throws KeyError Tarang.getint(sec, "nope")
        @test Tarang.getint(sec, "nope"; default=7) == 7
    end

    # ------------------------------------------------------------------
    # getfloat
    # ------------------------------------------------------------------
    @testset "getfloat" begin
        sec = Tarang.ConfigSection(Dict{String,Any}(
            "fval" => 3.14,
            "ival" => 10,
            "sval" => "2.718",
            "sbad" => "xyz",
        ), "floats")

        @test Tarang.getfloat(sec, "fval") == 3.14
        @test Tarang.getfloat(sec, "ival") == 10.0
        @test Tarang.getfloat(sec, "sval") == 2.718

        @test_throws ArgumentError Tarang.getfloat(sec, "sbad")
        @test_throws KeyError Tarang.getfloat(sec, "nope")
        @test Tarang.getfloat(sec, "nope"; default=0.0) == 0.0
    end

    # ------------------------------------------------------------------
    # getstring
    # ------------------------------------------------------------------
    @testset "getstring" begin
        sec = Tarang.ConfigSection(Dict{String,Any}(
            "s" => "hello",
            "n" => 42,
            "nil" => nothing,
        ), "strings")

        @test Tarang.getstring(sec, "s") == "hello"
        @test Tarang.getstring(sec, "n") == "42"

        # nothing value with no default throws
        @test_throws KeyError Tarang.getstring(sec, "nil")
        # nothing value with default returns default
        @test Tarang.getstring(sec, "nil"; default="fallback") == "fallback"

        @test_throws KeyError Tarang.getstring(sec, "nope")
        @test Tarang.getstring(sec, "nope"; default="def") == "def"
    end

    # ------------------------------------------------------------------
    # getlist
    # ------------------------------------------------------------------
    @testset "getlist" begin
        sec = Tarang.ConfigSection(Dict{String,Any}(
            "vec" => [1, 2, 3],
            "csv" => "a, b, c",
            "empty" => "",
            "scalar" => 42,
        ), "lists")

        @test Tarang.getlist(sec, "vec") == [1, 2, 3]
        @test Tarang.getlist(sec, "csv") == ["a", "b", "c"]
        @test Tarang.getlist(sec, "empty") == String[]
        @test Tarang.getlist(sec, "scalar") == [42]

        @test_throws KeyError Tarang.getlist(sec, "nope")
        @test Tarang.getlist(sec, "nope"; default=Int[]) == Int[]
    end

    # ------------------------------------------------------------------
    # Config object -- creation and defaults
    # ------------------------------------------------------------------
    @testset "Config defaults" begin
        cfg = Tarang.Config()
        @test haskey(cfg, "parallelism")
        @test haskey(cfg, "transforms")
        @test haskey(cfg, "logging")
        @test haskey(cfg, "debug")
        @test haskey(cfg, "memory")
        @test haskey(cfg, "solvers")

        # Section access returns ConfigSection
        sec = cfg["parallelism"]
        @test isa(sec, Tarang.ConfigSection)
        @test haskey(sec, "TRANSPOSE_LIBRARY")

        # show
        s = sprint(show, cfg)
        @test occursin("Config(", s)
    end

    # ------------------------------------------------------------------
    # get_value / set_value!
    # ------------------------------------------------------------------
    @testset "get_value and set_value!" begin
        cfg = Tarang.Config()

        @test Tarang.get_value(cfg, "debug", "ENABLED") == false
        @test Tarang.get_value(cfg, "nonexistent", "KEY"; default=:miss) == :miss

        Tarang.set_value!(cfg, "debug", "ENABLED", true)
        @test Tarang.get_value(cfg, "debug", "ENABLED") == true
        @test cfg._modified == true

        # set_value! creates section if missing
        Tarang.set_value!(cfg, "custom", "FOO", "bar")
        @test Tarang.get_value(cfg, "custom", "FOO") == "bar"
    end

    # ------------------------------------------------------------------
    # add_section!
    # ------------------------------------------------------------------
    @testset "add_section!" begin
        cfg = Tarang.Config()
        Tarang.add_section!(cfg, "newsec", Dict{String,Any}("x" => 1))
        @test haskey(cfg, "newsec")
        @test cfg["newsec"]["x"] == 1

        # Merging into existing section
        Tarang.add_section!(cfg, "newsec", Dict{String,Any}("y" => 2))
        @test cfg["newsec"]["x"] == 1
        @test cfg["newsec"]["y"] == 2
    end

    # ------------------------------------------------------------------
    # reset_config!
    # ------------------------------------------------------------------
    @testset "reset_config!" begin
        cfg = Tarang.Config()
        Tarang.set_value!(cfg, "debug", "ENABLED", true)
        @test Tarang.get_value(cfg, "debug", "ENABLED") == true

        Tarang.reset_config!(cfg)
        @test Tarang.get_value(cfg, "debug", "ENABLED") == false
        @test cfg._modified == false
    end

    # ------------------------------------------------------------------
    # TOML round-trip (save then load)
    # ------------------------------------------------------------------
    @testset "TOML save and load" begin
        cfg = Tarang.Config()
        Tarang.set_value!(cfg, "solvers", "MAX_ITERATIONS", 500)

        tmpfile = tempname() * ".toml"
        try
            @test Tarang.save_config(cfg, tmpfile) == true
            @test isfile(tmpfile)

            cfg2 = Tarang.Config()
            @test Tarang.load_config!(cfg2, tmpfile) == true
            @test Tarang.get_value(cfg2, "solvers", "MAX_ITERATIONS") == 500
        finally
            isfile(tmpfile) && rm(tmpfile)
        end
    end

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @testset "validate_config" begin
        # Valid config passes
        @test Tarang.validate_config(Dict{String,Any}()) == true

        # Invalid FFTW rigor
        @test_throws ArgumentError Tarang.validate_config(
            Dict("transforms-fftw" => Dict("PLANNING_RIGOR" => "INVALID")))

        # Invalid log level
        @test_throws ArgumentError Tarang.validate_config(
            Dict("logging" => Dict("LEVEL" => "VERBOSE")))

        # Invalid transpose library
        @test_throws ArgumentError Tarang.validate_config(
            Dict("parallelism" => Dict("TRANSPOSE_LIBRARY" => "BADLIB")))

        # Dealiasing factor < 1
        @test_throws ArgumentError Tarang.validate_config(
            Dict("transforms" => Dict("DEFAULT_DEALIASING" => 0.5)))

        # Negative solver tolerance
        @test_throws ArgumentError Tarang.validate_config(
            Dict("solvers" => Dict("DEFAULT_TOLERANCE" => -1.0)))
    end

    # ------------------------------------------------------------------
    # Environment variable overrides
    # ------------------------------------------------------------------
    @testset "apply_env_overrides!" begin
        cfg = Tarang.Config()

        # TARANG_LOG_LEVEL
        withenv("TARANG_LOG_LEVEL" => "DEBUG") do
            Tarang.apply_env_overrides!(cfg)
            @test Tarang.get_value(cfg, "logging", "LEVEL") == "DEBUG"
        end

        # TARANG_DEBUG
        Tarang.reset_config!(cfg)
        withenv("TARANG_DEBUG" => "true") do
            Tarang.apply_env_overrides!(cfg)
            @test Tarang.get_value(cfg, "debug", "ENABLED") == true
            @test Tarang.get_value(cfg, "logging", "LEVEL") == "DEBUG"
        end

        # TARANG_PROFILE_DIR
        Tarang.reset_config!(cfg)
        withenv("TARANG_PROFILE_DIR" => "/tmp/profiles") do
            Tarang.apply_env_overrides!(cfg)
            @test Tarang.get_value(cfg, "profiling", "PROFILE_DIRECTORY") == "/tmp/profiles"
        end

        # OMP_NUM_THREADS
        Tarang.reset_config!(cfg)
        withenv("OMP_NUM_THREADS" => "8") do
            Tarang.apply_env_overrides!(cfg)
            @test Tarang.get_value(cfg, "threading", "OMP_NUM_THREADS") == 8
        end
    end

    # ------------------------------------------------------------------
    # load_config! with missing file
    # ------------------------------------------------------------------
    @testset "load_config! missing file" begin
        cfg = Tarang.Config()
        @test Tarang.load_config!(cfg, "/nonexistent/path/config.toml") == false
    end

    # ------------------------------------------------------------------
    # Convenience helpers (global config)
    # ------------------------------------------------------------------
    @testset "Convenience functions" begin
        @test isa(Tarang.get_log_level(), String)
        @test isa(Tarang.get_fftw_rigor(), String)
        @test isa(Tarang.get_default_tolerance(), Number)
        @test isa(Tarang.get_dealiasing_factor(), Number)
        @test isa(Tarang.get_thread_count(), Integer)
        @test isa(Tarang.is_debug_enabled(), Bool)
        @test isa(Tarang.is_mpi_enabled(), Bool)
    end
end

println("All configuration tests passed!")
