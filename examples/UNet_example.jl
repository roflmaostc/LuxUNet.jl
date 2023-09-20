### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 5feb65dc-57ee-11ee-333f-6104bea77a76
begin
	using Pkg
	Pkg.activate(".")
	using Revise
end

# ╔═╡ 439f204a-055d-497d-a7b2-1404aea65dfd
using Lux, Random

# ╔═╡ 7bb399cf-86ec-4e9a-8b65-9627af866883
using LuxUNet

# ╔═╡ f42fefa4-5b6a-4f58-ac7a-490b19730649
using Optimisers

# ╔═╡ 983f203a-b37d-4610-bc9f-b9d0a330eaa6
rng = MersenneTwister()

# ╔═╡ 719ca3e8-976c-42b0-9cea-b53f9b56e431
Random.seed!(rng, 42)

# ╔═╡ da17b6e3-0286-4531-8ced-b3cbf80beb15
img = randn(rng, Float32, (20, 20, 1, 1));

# ╔═╡ 6953bb5c-5ab9-4504-a2aa-6ffd9df9c832
cb = LuxUNet.ConvBlock(1,2)

# ╔═╡ 77972415-a59c-4dec-a4f7-cda9d6f1a1f0
eb = LuxUNet.EncoderBlock(1,2)

# ╔═╡ 9cac3bee-670a-4530-8a22-cfc25f9bcf73
opt = Adam(0.03f0)

# ╔═╡ 324a1455-3e59-46f1-bdaf-c560fdf500ba
tstate = Lux.Training.TrainState(rng, cb, opt)

# ╔═╡ 8eb9a180-4164-447c-9cec-572c72660482
Lux.apply(cb, img, tstate.parameters, tstate.states)

# ╔═╡ d76ed330-009e-4af0-80d3-7f8308ef086b
tstate_eb = Lux.Training.TrainState(rng, eb, opt)

# ╔═╡ 74232991-a3d0-4132-a394-85b1cb09206b
Lux.apply(eb, img, tstate_eb.parameters, tstate_eb.states)

# ╔═╡ Cell order:
# ╠═5feb65dc-57ee-11ee-333f-6104bea77a76
# ╠═439f204a-055d-497d-a7b2-1404aea65dfd
# ╠═7bb399cf-86ec-4e9a-8b65-9627af866883
# ╠═f42fefa4-5b6a-4f58-ac7a-490b19730649
# ╠═983f203a-b37d-4610-bc9f-b9d0a330eaa6
# ╠═719ca3e8-976c-42b0-9cea-b53f9b56e431
# ╠═da17b6e3-0286-4531-8ced-b3cbf80beb15
# ╠═6953bb5c-5ab9-4504-a2aa-6ffd9df9c832
# ╠═77972415-a59c-4dec-a4f7-cda9d6f1a1f0
# ╠═9cac3bee-670a-4530-8a22-cfc25f9bcf73
# ╠═324a1455-3e59-46f1-bdaf-c560fdf500ba
# ╠═8eb9a180-4164-447c-9cec-572c72660482
# ╠═d76ed330-009e-4af0-80d3-7f8308ef086b
# ╠═74232991-a3d0-4132-a394-85b1cb09206b
