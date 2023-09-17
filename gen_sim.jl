# -*- coding: utf-8 -*-
using GeoStats #version 0.25.3
using Random
using CSV

mkdir("data")

dirc = [1, 4, 8, 12] # directory name
sacr = [1., 4., 8., 12.] # spatial autocorrelation range
ext = 60 # spatial extent
sim = 100 # number of simulations
ntr = 1800 # number of training points
nte = 500 # number of test points
thrs0 = 0.001 # for spatial sampling
spr = 30. # for spatial sampling

Random.seed!(42)

for (f,r) in zip(dirc,sacr)
	mkdir(string("./data/sac",f))
    # generate spatial process - Table 3
    grid = CartesianGrid(ext, ext)
    γ1 = GaussianVariogram(range=r, sill=1., nugget=0.) 
    γ2 = GaussianVariogram(range=r, sill=1., nugget=0.) 
    γ3 = GaussianVariogram(range=spr, sill=1., nugget=0.)
    vars = (:X1=>Float64, :X2=>Float64, :X3=>Float64)
    prob = SimulationProblem(grid, vars, sim)
    solv = FFTGS(:X1=>(mean=0., variogram=γ1), :X2=>(mean=0., variogram=γ2), :X3=>(mean=0., variogram=γ3))
    sol = solve(prob, solv)

    # sampling points
    for i in range(1,length=sim)
        # case 1: uniform sampling - extract tr & te points from the same landscape.
        tr = sample(sol[i], UniformSampling(ntr))
        tr_file = string("./data/sac",f,"/c1_tr", "_", i, ".csv")
        te = sample(sol[i], UniformSampling(nte))
        te_file = string("./data/sac",f,"/c1_te", "_", i, ".csv")
        CSV.write(tr_file, tr)
        CSV.write(te_file, te)

        # case 3: spatial sampling - extract tr & te points from the same landscape.
        x3 = asarray(sol[i], :X3)
        x3_flat = reshape(x3, :)
        de = maximum(x3_flat) - minimum(x3_flat)
        x3_scale = (x3_flat .- minimum(x3_flat)) ./ de 
        thrs1 = mean(x3_scale) + 0.1
        thrs2 = 1.2 - mean(x3_scale)
        x3_weight_tr = [j <= thrs1 ? thrs0 : j for j in x3_scale]
        tr = sample(sol[i], WeightedSampling(ntr, x3_weight_tr; replace=false))

        x3_weight_rev = broadcast(-, 1, x3_scale)
        x3_weight_te = [k <= thrs2 ? thrs0 : k for k in x3_weight_rev]
        te = sample(sol[i], WeightedSampling(nte, x3_weight_te; replace=false))

        tr_file = string("./data/sac",f,"/c3_tr", "_", i, ".csv")
        te_file = string("./data/sac",f,"/c3_te", "_", i, ".csv")
        CSV.write(tr_file, tr)
        CSV.write(te_file, te)
    end
end


for (f,r) in zip(dirc,sacr)
    # generate spatial process - Table 3
    grid = CartesianGrid(ext, ext)
    γ4 = GaussianVariogram(range=r, sill=0.5, nugget=0.) 
    γ5 = GaussianVariogram(range=r, sill=0.5, nugget=0.) 
    γ6 = GaussianVariogram(range=r, sill=1.0, nugget=0.)
    γ7 = GaussianVariogram(range=r, sill=1.0, nugget=0.)
    vars = (:X1=>Float64, :X2=>Float64, :X3=>Float64, :X4=>Float64, :X5=>Float64, :X6=>Float64)
    prob = SimulationProblem(grid, vars, sim)
    # 3 configurations for SI+CS - inside: X1, X2; partial: X3, X4; outside: X5, X6
    solv = FFTGS(:X1=>(mean=0., variogram=γ4), :X2=>(mean=0., variogram=γ5), :X3=>(mean=1., variogram=γ6), :X4=>(mean=1., variogram=γ7), :X5=>(mean=7., variogram=γ6), :X6=>(mean=7., variogram=γ7))
    sol = solve(prob, solv)
    
    # sampling points
    for i in range(1,length=sim)
        # case 4: extract tr & te points from different landscapes.
        te = sample(sol[i], UniformSampling(nte))
        te_file = string("./data/sac",f,"/c4_te", "_", i, ".csv")
        CSV.write(te_file, te)
    end
end
