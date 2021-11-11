module GZK_sim

using CSV
using DataFrames
using Random
using ThreadsX

include("./pions.jl")
include("./protons.jl")

using .PionFunctions
using .ProtonFunctions

export propagate

"""
    'propagate(z_dist, E_dist)'

Propagates protons to the Earth (z=0), calculating the energy they lose and the neutrinos they create along the way. This is the main loop of the simulation.

Inputs:
- z_dist = array of the initial redshift (z) of each proton
- E_dist = array of the initial energy (log10(E/eV)) of each proton 

Output:
Dataframe with columns for the event #, injection energy, injection redshift, arriving proton energy, # of arriving neutrinos, and an array of the arriving neutrino energies.
"""
function propagate(z_dist::AbstractArray{T1}, E_dist::AbstractArray{T2}, seed = false) where {T1<:Real, T2<:Real}

    if seed != false
        rng = MersenneTwister(seed)
    else
        rng = MersenneTwister()
    end
    
    # load in the pγ cross-section data
    data = CSV.read("./data/p-gamma_cross_section.csv", DataFrame) #Source: M. Tanabashi et al. (Particle Data Group), Phys. Rev. D 98, 030001 (2018).
    ϵ = data[!,1]*1e3 # photon energy in MeV
    σ = data[!,2]*1e-13 # cross-section in nanometer^2

    primaries = [E_dist z_dist]

    # preallocate the output vectors. Fill it with "missing" so we can tell if something went wrong by looking at the output CSV.
    event = Vector{Union{Missing, Int}}(missing, size(primaries,1))
    injection_energy = Vector{Union{Missing, Float64}}(missing, size(primaries,1))
    redshift = Vector{Union{Missing, Float64}}(missing, size(primaries,1))
    proton_energies = Vector{Union{Missing, Float64}}(missing, size(primaries,1))
    num_of_neutrinos = Vector{Union{Missing, Float64}}(missing, size(primaries,1))
    neutrino_energies = Vector{Union{Missing, Array}}(missing, size(primaries,1))

    # Create a separate RNG for each thread
    rng_thread = map(i->MersenneTwister(rand(rng,UInt32)),1:Threads.nthreads())
    
    # loop over each injected primary. 
    #Threads.@threads for i in (1:size(primaries,1))
    ThreadsX.foreach(1:size(primaries,1), basesize=ceil(Int, size(primaries,1)/Threads.nthreads())) do i
    
        mp = 938.272e6 # proton mass, eV
        Γ_initial = (big(10)^primaries[i,:][1])/mp # the initial Lorentz factor of the proton
        z = LinRange(primaries[i,:][2],0,100) # z ranges from the injection redshift to Earth (z=0)
        initial_rate = interaction_rate(Γ_initial, z[1], ϵ, σ)
        u = rand(rng_thread[Threads.threadid()]) # this random number will be used to decide when the interaction takes place
        p_initial = 1 # initial interaction probability is 0
        
        # define arrays to store values
        proton = [Γ_initial,initial_rate,p_initial,u]
        neutrinos = []

        # loop over each step in redshift
        for j in (2:size(z,1))
    
            proton[1] = adiabatic_energy_loss(proton[1], z[j-1], z[j])
            rate = interaction_rate(proton[1], z[j], ϵ, σ)
            p = survival_probability(proton[3], proton[2], z[j-1], rate, z[j])
            proton[3] = p
            if p < proton[4] # if p < u then the interaction takes place
                if rand(rng_thread[Threads.threadid()], 1:3) == 1 # neutral pions have a 1/3 chance of being produced
                    # neutral pion production creates a new proton and no neutrinos
                    proton = neutral_pion_production(proton[1], z[j], rate, rng_thread[Threads.threadid()])
                else # charged pions have a 2/3 chance of being produced
                    # charged pion production creates a new proton and 4 neutrinos
                    proton, neutrino1, neutrino2, neutrino3, neutrino4 = charged_pion_production(proton[1], z[j], rate, rng_thread[Threads.threadid()])
                    append!(neutrinos, [neutrino1, neutrino2, neutrino3, neutrino4])
                end
            else
                proton[2] = rate # if the proton doesn't interact simply update the interaction rate and loop again
            end
        end
            
        # loop over each neutrino that was created
        for k in (1:size(neutrinos,1))
            # neutrinos do not undergo any energy loss as they propagate except from the adiabatic expansion of the universe.
            # the energy of a neutrino arriving to Earth is therefore E/(1+z), where E and z are the energy and redshift at which the neutrino was created.
            neutrinos[k][1] = neutrinos[k][1]/(1+neutrinos[k][2]) 
        end
        
        # fill in the output vectors
        event[i] = i
        injection_energy[i] = primaries[i,:][1]
        redshift[i] = primaries[i,:][2]
        proton_energies[i] = log10(proton[1]*mp)
        num_of_neutrinos[i] = size(neutrinos,1)
        neutrino_energies[i] = log10.([convert(Float64,x[1]) for x in neutrinos].*1e6)
            
    end

    # construct the dataframe
    df = DataFrame(Event=event, Injection_Energy=injection_energy, Injection_Z=redshift, Arriving_Proton_Energy=proton_energies, Arriving_Neutrinos=num_of_neutrinos, 
    Neutrino_Energies=neutrino_energies)

    return df

end


end