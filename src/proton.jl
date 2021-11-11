module ProtonFunctions

using QuadGK
using Trapz

export adiabatic_energy_loss, interaction_rate, hubbles_law, survival_probability

"""
    `adiabatic_energy_loss(lorentz_initial, z_initial, z_final)`

Calculates the Lorentz factor of a particle after it propagates from a redshift z_initial to a redshift z_final.
The particle loses energy due to the adiabatic expansion of the universe.

Based on eq. (2-4) from arXiv:1505.01347 https://arxiv.org/abs/1505.01347
"""
function adiabatic_energy_loss(lorentz_initial::Real, z_initial::Real, z_final::Real)
    integral, err = quadgk(z -> 1/(1+z), z_initial, z_final) # numerical integration using QuadGK.jl
    energy_loss = ℯ^(-integral)
    return lorentz_initial/energy_loss
end


"""
    `interaction_rate(Γ, ϵ, σ)`
Calculates the interaction rate of protons with photons for a given proton Lorentz factor, redshift, photon energy, and interaction cross section.

Inputs:
- Γ: proton Lorentz factor
- z: redshift
- ϵ: photon energy (in the proton rest frame) (eV)
- σ: pγ interaction cross-section (micrometer^2)

Source: Eq. (10.9) from Cosmic Rays and Particle Physics, 2nd Edition, by Thomas K. Gaisser, Ralph Engel, and Elisa Resconi.
"""
function interaction_rate(Γ::Real, z::Real, ϵ::Vector{T1}, σ::Vector{T2}) where {T1<:Number, T2<:Number}
    c = (2.99792e8)*1e9 # speed of light, nanometer/s
    kT = 0.234822*1e-9 # boltzmann constant * z=0 CMB temperature, MeV
    ħc = .197327*1e-3 # reduced planck constant * speed of light, MeV * nanometer
    M = @. ϵ*σ*(-log(1-ℯ^(-ϵ/(2*Γ*kT))))
    # trapezoidal integration of M over the array of ϵ
    I = trapz(ϵ, M) 
    @assert I >= 0.0 # An interaction rate will never be negative
    rate_at_zero_redshift = (c*kT)/((2*π^2)*(ħc^3)*(Γ^2)) * I
    return rate_at_zero_redshift*(1+z)^3
end


"""
    'hubbles_law(z)'

Calculates dt/dz at the input redshift z. Source: eq. (3) from arXiv:1505.01347 https://arxiv.org/abs/1505.01347
"""
function hubbles_law(z::Real)
    H0 = 2.198e-18 # Hubble constant in s^-1
    Ωm = 0.3 # matter density parameter
    ΩΛ = 0.7 # dark energy density parameter
    return -1/(H0*(1+z)*sqrt((1+z^3)*Ωm + ΩΛ))
end


"""
    'survival_probability(p_prev, rate_prev, z_prev, rate, z)'

The probability that the proton has not undergone a reaction at each redshift step, approximated
using the trapezoidal rule. Source: Eq. (7) arXiv:1505.01347 https://arxiv.org/abs/1505.01347

Inputs:
- p_prev = the survival probability at the previous redshift step
- rate_prev = the interaction rate at the previous redshift step
- z_prev = the redshift at the previous redshift step
- rate = the interaction rate at the current redshift step
- z = the redshift at the current redshift step
"""
function survival_probability(p_prev::Real, rate_prev::Real, z_prev::Real, rate::Real, z::Real)
    @assert p_prev >= 0 # probabilites cannot be negative
    logp = log(p_prev) - (rate_prev*hubbles_law(z_prev) + rate*hubbles_law(z))*(0.5)*(z - z_prev)
    return ℯ^logp
end

end