module PionFunctions

using Random

export neutral_pion_production, charged_pion_production

"""
    'neutral_pion_production(Γ, z, rate)'

p + γ -> p + π(0). Neutral pions decay into photons, which we do not track in this simulation. The interaction does however create a proton of lower energy, which we do wish to track.

Inputs:
- Γ = proton Lorentz factor
- z = redshift
- rate = pγ interaction rate at the current redshift 
- rng = the MersenneTwister RNG

Equations are derived in Ch. 10 of Cosmic Rays and Particle Physics, 2nd Edition, by Thomas K. Gaisser, Ralph Engel, and Elisa Resconi.
"""
function neutral_pion_production(Γ::Real, z::Real, rate::Real, rng::MersenneTwister)
    @assert Γ >= 1
    @assert z >= 0
    mp = 938.272 # proton mass, MeV
    mπ = 134.9768 # neutral pion mass, MeV
    # the photon and proton can impact at any angle
    cos_theta_impact = rand(rng, -1:0.0001:0.9999)
    photon_energy = CMB_spectrum_rand(z, Γ, mπ, cos_theta_impact, rng)
    # COM energy
    s = mp^2 + 2*mp*Γ * photon_energy * (1 - sqrt(1-(1/Γ^2))*cos_theta_impact) 
    pion_energy_COM = (s - mp^2 + mπ^2)/(2*sqrt(s))
    @assert pion_energy_COM >= mπ
    pion_momentum_COM = sqrt(pion_energy_COM^2 - mπ^2)
    # the products can scatter at any angle
    cos_theta_scatter = rand(rng, -1:0.0001:1)
    pion_energy = mp*Γ*(pion_energy_COM + pion_momentum_COM*cos_theta_scatter)/sqrt(s)
    proton_energy = mp*Γ - pion_energy
    @assert pion_energy >= 0
    @assert proton_energy >= 0
    proton = [proton_energy/mp, rate, 1, rand(rng)] # replace old proton with new proton
    
    return proton
end


"""
    'charged_pion_production(Γ, z, rate)'

p + γ -> n + π(+). The free neutron will decay to create a proton, an electron, and an anti-electron neutrino. The charged pion will decay to form a anti-muon and a muon neutrino. The anti-muon 
will then decay to a positron, electron neutrino, and anti-muon neutrino. Overall this interaction therefore produces 4 neutrinos, as well as a proton that continues to propagate.

Inputs:
- Γ = proton Lorentz factor
- z = redshift
- rate = pγ interaction rate at the current redshift 
- rng = the MersenneTwister RNG

Equations are derived in Ch. 10 of Cosmic Rays and Particle Physics, 2nd Edition, by Thomas K. Gaisser, Ralph Engel, and Elisa Resconi.
"""
function charged_pion_production(Γ::Real, z::Real, rate::Real, rng::MersenneTwister)
    @assert Γ >= 1
    @assert z >= 0
    mp = 938.272 # proton mass, MeV
    mπ = 139.5704 # charged pion mass, MeV
    # the photon and proton can impact at any angle
    cos_theta_impact = rand(rng, -1:0.0001:0.9999)
    photon_energy = CMB_spectrum_rand(z, Γ, mπ, cos_theta_impact, rng)
    # COM energy
    s = mp^2 + 2*mp*Γ * photon_energy * (1 - sqrt(1-(1/Γ^2))*cos_theta_impact) 
    pion_energy_COM = (s - mp^2 + mπ^2)/(2*sqrt(s))
    @assert pion_energy_COM >= mπ
    pion_momentum_COM = sqrt(pion_energy_COM^2 - mπ^2)
    # the products can scatter at any angle
    cos_theta_scatter = rand(rng, -1:0.0001:1)
    pion_energy = mp*Γ*(pion_energy_COM + pion_momentum_COM*cos_theta_scatter)/sqrt(s)
    neutron_energy = mp*Γ - pion_energy
    @assert pion_energy >= 0
    @assert neutron_energy >= 0
    proton, neutrino1 = neutron_decay(neutron_energy, rate, z, rng)
    neutrino2, neutrino3, neutrino4 = charged_pion_decay(pion_energy, z, rng)
    
    return proton, neutrino1, neutrino2, neutrino3, neutrino4 
end

"""
    'CMB_spectrum_rand(z, Γ, mπ, cos_impact_angle)'

Samples a photon energy (in the CMB rest frame) at redshift z and above the pγ interaction threshold from the blackbody spectrum.

Inputs:
- z = redshift
- Γ = Lorentz factor of the interacting proton
- mπ = pion mass in MeV (neutral and charged pions have different masses)
- cos_impact_angle = cosine of the angle between the interacting proton and photon
- rng = the MersenneTwister RNG
"""
function CMB_spectrum_rand(z::Real, Γ::Real, mπ::Real, cos_impact_angle::Real, rng::MersenneTwister)
    mp = 938.272 # proton mass, MeV
    kT = 0.2348*1e-3 # boltzmann constant * z=0 CMB temperature, MeV
    ħc = .197327*1e-9 # reduced planck constant * speed of light, MeV * nanometer
    threshold = (mπ^2 + 2*mp*mπ)/(2*mp*Γ*(1-sqrt(1-(1/Γ^2))*cos_impact_angle)) # we've already determined that the interaction has happened
    energy = LinRange(threshold, 1e-2, 1000) # this seems to be a good range to sample the spectrum
    spectrum = @. (energy^2)/((π^2)*(ħc)^3*(ℯ^(energy/(kT*(1+z)))-1)) # Planck's Law
    pdf = spectrum/sum(spectrum)
    normalized_pdf = pdf/sum(pdf)
    cdf = cumsum(normalized_pdf)
    cdf_idx = searchsortedfirst(cdf, rand(rng))
    return energy[cdf_idx]
end


"""
    'neutron_decay(neutron_energy, rate)'

A free neutron will quickly decay: n -> p + e(-) + anti-ν_e. Since this is a 3-body decay the energy of the products are not guaranteed. The electron energy is therefore sampled from an energy spectrum.
The neutrino energy is then calculated using the conservation of energy, assuming that the proton receives minimal kinetic energy due to its much heavier mass. We then track the neutrino and the proton
through the rest of the sim.

Based on the procedure done in arXiv:1505.01347 https://arxiv.org/abs/1505.01347
"""
function neutron_decay(neutron_energy::Real, rate::Real, z::Real, rng::MersenneTwister)
    me = 0.510999 # electron mass, MeV
    mn = 939.565 # neutron mass, MeV
    Q = 0.782343 # Q value of free neutron decay, MeV
    Γ = neutron_energy/mn
    electron_energy = ndecay_electron_energy(me, Q, rng) # rest frame
    @assert electron_energy >= me
    neutrino_energy_restframe = Q - (electron_energy - me)
    cos_scattering_angle = rand(rng, -1:0.0001:1) # assume scattering is equally likely in all directions
    neutrino_energy = Γ*neutrino_energy_restframe*(1-cos_scattering_angle)
    @assert neutrino_energy >= 0
    neutrino1 = [neutrino_energy,z] # add the neutrino energy and the redshift at which it was created to the neutrino array
    # the resulting proton has approximately the same energy as the parent neutron
    proton = [Γ, rate, 1, rand(rng)] # replace old proton with new proton
    
    return proton, neutrino1
end


"""
    'electron_energy_spectrum(electron_energy, m, Q)'

The probability distribution function (PDF) of electron energy possible during free neutron decay. Source: arXiv:1505.01347 https://arxiv.org/abs/1505.01347
"""
function electron_energy_spectrum(electron_energy::Real, m::Real, Q::Real)
    @assert (electron_energy^2 - m^2) >= 0
    return sqrt(electron_energy^2 - m^2)*electron_energy*(Q - (electron_energy - m))^2
end


"""
    'electron_energy()'

Returns a random electron energy (in the rest frame) resulting from free neutron decay.

Inputs:
- m = electron mass 
- Q = Q-value of free neutron decay
- rng = the MersenneTwister RNG
"""
function ndecay_electron_energy(m::Real, Q::Real, rng::MersenneTwister)
    # the range of energies the electron is allowed to have by energy conservation:
    energies = LinRange(m, m+Q, 1000)
    pdf = electron_energy_spectrum.(energies, m, Q)
    normalized_pdf = pdf/sum(pdf)
    cdf = cumsum(normalized_pdf)
    cdf_idx = searchsortedfirst(cdf, rand(rng))
    return energies[cdf_idx]
end


"""
    'charged_pion_decay(pion_energy)'

A positively-charged pion will decay to form a anti-muon and a muon neutrino. The anti-muon will then go on to also decay.

Based on the procedure done in arXiv:1505.01347 https://arxiv.org/abs/1505.01347
"""
function charged_pion_decay(pion_energy::Real, z::Real, rng::MersenneTwister)
    mμ = 105.6584 # muon mass, MeV
    mπ = 139.5704 # charged pion mass, MeV
    muon_max_energy = (1 - mμ^2/mπ^2)*pion_energy
    muon_energy = rand(rng, 0:muon_max_energy) # possible muon energy is uniform from 0 to (1 - mμ^2/mπ^2)*pion_energy
    neutrino_energy = pion_energy - muon_energy
    @assert neutrino_energy >= 0
    neutrino2 = [neutrino_energy,z] # add the neutrino energy and the redshift at which it was created to the neutrino array
    neutrino3, neutrino4 = muon_decay(muon_energy, z, rng)
    
    return neutrino2, neutrino3, neutrino4
end


"""
    'muon_decay(muon_energy)'

A muon (or anti-muon) will decay to create an electron (positron) and two neutrinos. Here we calculate two neutrino energies that can occur from the decay of a muon with energy muon_energy, and then
append the neutrino array.
"""
function muon_decay(muon_energy::Real, z::Real, rng::MersenneTwister)
    mμ = 105.6584 # muon mass, MeV
    me = 0.510999 # electron mass, MeV
    max_energy = mμ/2 - (me^2)/2mμ # max energy a neutrino can have from muon decay in the rest frame
    # neutrinos have a random energy uniformly from 0 to the max energy
    ν1_restframe_energy = rand(rng, 0:0.0001:max_energy)
    ν2_restframe_energy = rand(rng, 0:0.0001:max_energy)
    electron_restframe_energy = mμ - ν1_restframe_energy - ν2_restframe_energy
    # make sure conservation of energy is not violated
    if electron_restframe_energy < me
        # recalculate values until they are ok
        while electron_restframe_energy < me 
            ν1_restframe_energy, ν2_restframe_energy = rand(rng, 0:0.0001:max_energy, 2)
            electron_restframe_energy = mμ - ν1_restframe_energy - ν2_restframe_energy
        end
    end
    # neutrinos are nearly massless, so E=p
    ν1_momentum = ν1_restframe_energy
    ν2_momentum = ν2_restframe_energy
    electron_momentum = sqrt(electron_restframe_energy^2 - me^2)
    # make sure conservation of momentum is not violated
    if ν1_momentum > ν2_momentum + electron_momentum || ν2_momentum > ν1_momentum + electron_momentum || electron_momentum > ν1_momentum + ν2_momentum
        # recalculate values until they are ok
        while electron_restframe_energy < me || ν1_momentum > ν2_momentum + electron_momentum || ν2_momentum > ν1_momentum + electron_momentum || electron_momentum > ν1_momentum + ν2_momentum
            ν1_restframe_energy, ν2_restframe_energy = rand(rng, 0:0.0001:max_energy, 2)
            electron_restframe_energy = mμ - ν1_restframe_energy - ν2_restframe_energy
            ν1_momentum = ν1_restframe_energy
            ν2_momentum = ν2_restframe_energy
            if electron_restframe_energy >= me
                electron_momentum = sqrt(electron_restframe_energy^2 - me^2)
            else
                electron_momentum = NaN # It will be recalculated anyways
            end
        end
    end
    # time to convert energies to lab frame
    theta12 = acos((electron_momentum^2 - ν1_momentum^2 - ν2_momentum^2)/(2*ν1_momentum*ν2_momentum)) # the angle between the two neutrinos
    theta1 = rand(rng, 0:0.0001:2π) # the angle between the 1st neutrino and the line of sight (LoS) is isotropic
    phi = rand(rng, 0:0.0001:2π) # the angle between the 2nd neutrino and the plane containing the LoS and the 1st neutrino is isotropic
    theta2 = acos(cos(theta12)*cos(theta1) - sin(theta12)*sin(theta1)*cos(phi)) # the angle between the 2nd neutrino and the line of sight 
    ν1_energy = (muon_energy/mμ)*(ν1_restframe_energy + ν1_momentum*cos(theta1))
    ν2_energy = (muon_energy/mμ)*(ν2_restframe_energy + ν2_momentum*cos(theta2))
    
    @assert ν1_energy >= 0 
    @assert ν2_energy >= 0 

    # Add the two neutrinos to the neutrino array
    neutrino3 = [ν1_energy,z] # add the neutrino energy and the redshift at which it was created to the neutrino array
    neutrino4 = [ν2_energy,z] # add the neutrino energy and the redshift at which it was created to the neutrino array
    
    return neutrino3, neutrino4
end

end