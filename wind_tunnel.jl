using HDF5
using StaticArrays
using Statistics
using Random
using LinearAlgebra
using Printf
using Parameters

using PyPlot

const SOLAMASS = 1.989e+43
const PROTONMASS = 1.6726e-24
const XH = 0.71
const BOLTZMANN = 1.38e-16
const GAMMA = 5.0 / 3

const UnitMass_in_g = 1.989e+43
const UnitLength_in_cm = 3.08568e+21
const UnitVelocity_in_cm_per_s = 1e5

@with_kw struct Params{NDIM,T}

    #Lx::T   # box size in x-direction
    #Ly::T   # box size in y-direction

    boxsize::T # box size in y- and z-directions
    efac::T #box size in x-direction in units of boxsize

    Tc::T   # cloud temperature
    Th::T   # hot gas temperature

    nc::T   # cloud density
    nh::T   # hot gas density

    xc::T   # x coordinate for the cloud center in units of Lx
    Rc::T   # cloud radius
    vx::T   # relative velocity (in x-direction)

    Nc::Int   # number of particles in the spherical cloud


    seed::Int64 = 0
    filename::String = ""
end


function uniform_within_a_sphere(Np)
    x = randn(Np)
    y = randn(Np)
    z = randn(Np)
    r3d = @. sqrt(x^2 + y^2 + z^2)
    U = rand(Np).^(1/3)   #importance sampling
    pos = zeros(3,Np)
    @. x = x / r3d * U
    @. y = y / r3d * U
    @. z = z / r3d * U
    return vcat(x', y', z')
end

function generate_wind_tunnel(par::Params{NDIM,T}) where{NDIM,T}

    @unpack boxsize, efac, nc, nh, Tc, Th, xc, Rc, vx, Nc, filename, seed = par

    Ly=Lz=boxsize
    Lx=Ly*efac
    #efac = Int(round(Lx/Ly))   #elongation factor
    @show Lx,Ly,Lz

    cloudcenter = [xc*Lx,0.5*Ly,0.5*Lz]
    #cloudcenter = [Lx,Ly,Lz] .* 0.5

    χ = nc / nh    

    volume_box = Lx*Ly*Lz
    volume_cloud = 4*pi/3 * Rc^3


    XH = 0.71
    fac_rho2n = 404.76895856 * XH

    rho_c = nc / fac_rho2n  #gadget unit [1e10 M_sun/kpc^3]

    M_cloud = rho_c * volume_cloud
    
    m_gas = M_cloud / Nc   

    Nh_float = Nc/χ * (volume_box / volume_cloud)
    Ngrid = floor(Int, (Nh_float / efac)^(1/3) )


    Nh_uncut = Int(Ngrid^3 * efac)  #number of particles in the hot background medium (uniform grid) 
    dx = Ly / Ngrid  #cell size
     
    pos_h = zeros(3,Nh_uncut)

    n = 0
    for i in 1:Ngrid*efac, j in 1:Ngrid, k in 1:Ngrid
        pos_n = ( [i,j,k] .- 0.5 ) .* dx
        if sum( (pos_n .- cloudcenter).^2 ) > Rc^2   #outside the cloud
            n += 1
            pos_h[:,n] = pos_n
        end
    end
    Nh = n

    pos_h = pos_h[:,1:Nh]

    
    Random.seed!(seed);
    pos_c = uniform_within_a_sphere(Nc) .* Rc .+ cloudcenter

    #pos = pos_h
    pos = hcat(pos_h, pos_c)

    N_gas = size(pos,2)

    #vel_h = zeros(3,Nh)
    #vel_c = vcat(-vx .* ones(Nc)', zeros(2,Nc)) #cloud moves in the -x direction

    vel_h = vcat( vx .* ones(Nh)', zeros(2,Nh)) #wind moves in the +x direction
    vel_c = zeros(3,Nc)
    
    #vel = vel_h
    vel = hcat(vel_h, vel_c)
    
    @show Ngrid, Nh, Nc, Nh_uncut, dx

    @show χ, m_gas

    
    #fac = BOLTZMANN / (GAMMA-1.0) / PROTONMASS / UnitVelocity_in_cm_per_s^2

    #T_gas = vcat(ones(Nh) .* Th)
    T_gas = vcat(ones(Nh) .* Th, ones(Nc) .* Tc)

    mu = 0.62  #ionized gas
    fac_T2u = BOLTZMANN / ((GAMMA-1.0) * mu * PROTONMASS * UnitVelocity_in_cm_per_s^2)
    u_gas = T_gas .* fac_T2u

    @show u_gas[1], u_gas[end]

    id_gas = collect(Int32, 1:N_gas)
    m_gas = ones(N_gas) .* m_gas
    #hsml = ones(N_gas) .* dx   # will be re-calculated in Gadget, so we just guess some ballpark number

    # convert to Gadget code unit
    #m_gas .*= 1e-10   # 1e10 M_sun
    #pos .*= 1e-3   # kpc
    #hsml .*= 1e-3   # kpc

    if filename == ""
        filename = "ics_wind"
        filename *= "_Tc" * @sprintf("%.0e", Tc) *
                    "_Th" * @sprintf("%.0e", Th) *
                    "_nc" * @sprintf("%.0e", nc) *
                    "_nh" * @sprintf("%.0e", nh) *
                    "_vx" * @sprintf("%.0e", vx) *
                    "_Rc" * @sprintf("%.0e", Rc) *
                    "_Lx" * @sprintf("%.0e", Lx) *
                    "_Ly" * @sprintf("%.0e", Ly) *
                    "_Nc" * @sprintf("%.0e", Nc) *
                    ".hdf5"
    end

    ########## write to file ##########
    println("saving to file...")
    save_gadget_ics(filename, pos, vel, id_gas, m_gas, u_gas, N_gas, boxsize)
    println("done")

    return pos, vel, id_gas, m_gas, u_gas, N_gas, Ly
end

function save_gadget_ics(filename, pos, vel, id_gas, m_gas, u_gas, N_gas, boxsize)
    T = Float32

    fid=h5open(filename,"w")

    grp_head = create_group(fid,"Header");
    attributes(fid["Header"])["NumPart_ThisFile"]       = Int32[N_gas, 0, 0, 0, 0, 0]
    attributes(fid["Header"])["MassTable"]              = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    attributes(fid["Header"])["Time"]                   = 0.0
    attributes(fid["Header"])["Redshift"]               = 0.0
    attributes(fid["Header"])["Flag_Sfr"]               = 0
    attributes(fid["Header"])["Flag_Feedback"]          = 0
    attributes(fid["Header"])["NumPart_Total"]          = Int32[N_gas, 0, 0, 0, 0, 0]
    attributes(fid["Header"])["Flag_Cooling"]           = 0
    attributes(fid["Header"])["NumFilesPerSnapshot"]    = 1
    attributes(fid["Header"])["BoxSize"]                = boxsize
    attributes(fid["Header"])["Omega0"]                 = 0.27
    attributes(fid["Header"])["OmegaLambda"]            = 0.73
    attributes(fid["Header"])["HubbleParam"]            = 1.0
    attributes(fid["Header"])["Flag_StellarAge"]        = 0
    attributes(fid["Header"])["Flag_Metals"]            = 0
    attributes(fid["Header"])["NumPart_Total_HighWord"] = UInt32[0,0,0,0,0,0]
    attributes(fid["Header"])["flag_entropy_instead_u"] = 0
    attributes(fid["Header"])["Flag_DoublePrecision"]   = 0
    attributes(fid["Header"])["Flag_IC_Info"]           = 0
    #attributes(fid["Header"])["lpt_scalingfactor"] =

    grp_part = create_group(fid,"PartType0");
    h5write(filename, "PartType0/Coordinates"   , T.(pos))
    h5write(filename, "PartType0/Velocities"    , T.(vel))
    h5write(filename, "PartType0/ParticleIDs"   , id_gas)
    h5write(filename, "PartType0/Masses"        , T.(m_gas))
    h5write(filename, "PartType0/InternalEnergy", T.(u_gas))

    close(fid)
end


Rc=0.001
boxsize = Rc * 10 #Ly
efac = 5 #Lx = Ly * efac
nc=1e0
nh=1e-2
Tc=1e4
Th=1e6
xc=0.2
vx=200
Nc=100_000

#filename="ics_wind_N1e5_Rc0p1_Lx3Ly1.hdf5"
filename="ics_wind_N1e5_Rc0p001_Ly0p01_efac5.hdf5"

par = Params{3,Float64}(boxsize=boxsize, efac=efac, Rc=Rc, xc=xc, nc=nc, nh=nh, Tc=Tc, Th=Th, vx=vx, Nc=Nc, filename=filename);
pos, vel, id_gas, m_gas, u_gas, N_gas, Ly = generate_wind_tunnel(par);
clf()
plot(pos[1,:], pos[2,:], ".", ms=0.01)
#axis([0,Lx,0,Ly])
