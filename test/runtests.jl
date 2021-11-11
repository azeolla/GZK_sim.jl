using Test
using CSV
using DataFrames
using Random

include("../src/pions.jl")
include("../src/proton.jl")

using .ProtonFunctions
using .PionFunctions

data = CSV.read("../src/data/p-gamma_cross_section.csv", DataFrame)
ϵ = data[!,1]*1e3 # photon energy in MeV
σ = data[!,2]*1e-13 # cross-section in nanometer^2

mp = 938e6 # eV

@testset "Testing interaction_rate function" begin
    # interaction_rate below the threshold (5e19 eV) should be zero
    @test interaction_rate((1e16/mp), 0, ϵ, σ) ≈ 0 
    @test interaction_rate((1e17/mp), 0, ϵ, σ) ≈ 0  
    @test interaction_rate((1e18/mp), 0, ϵ, σ) ≈ 0 
    @test interaction_rate((1e19/mp), 0, ϵ, σ) ≈ 0  
end;

@testset "Testing adiabatic_energy_loss function" begin
    # after propagating to Earth, E = E/(1+z)
    @test adiabatic_energy_loss(1e10, 1, 0) ≈ (1e10)/2 
    @test adiabatic_energy_loss(1e10, 4, 0) ≈ (1e10)/5 
end;

@testset "Testing survival_probability function" begin
    # if there is no interaction the survival probability should be 1
    @test survival_probability(1, 0, 1, 0, 0) ≈ 1  
    # for a high rate the survival probability should be 0
    @test survival_probability(1, 1, 1, 1, 0) ≈ 0  
end;

# Testing assertions:

rng = MersenneTwister()

@testset "Testing neutral_pion_production function" begin 
    # low energy protons cannot create pions
    @test_throws AssertionError neutral_pion_production((1e10/mp), 0, 0, rng)
    @test_throws AssertionError neutral_pion_production((1e11/mp), 0, 0, rng)
    @test_throws AssertionError neutral_pion_production((1e12/mp), 0, 0, rng)
end;

@testset "Testing charged_pion_production function" begin 
    # low energy protons cannot create pions
    @test_throws AssertionError charged_pion_production((1e10/mp), 0, 0, rng)
    @test_throws AssertionError charged_pion_production((1e11/mp), 0, 0, rng)
    @test_throws AssertionError charged_pion_production((1e12/mp), 0, 0, rng)
end;