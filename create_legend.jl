using Pkg
Pkg.activate(".")
using Plots

include("my_theme.jl")
theme(:custom)

p_leg = plot(
    legend = :bottom,
    legendcolumns = 4,
    legendfontsize = 10,

    framestyle = :none,
    grid = false,
    showaxis = false,

    left_margin = 0Plots.mm,
    right_margin = 0Plots.mm,
    top_margin = 0Plots.mm,
    bottom_margin = 0Plots.mm,

    size=(900,60),

    # Make the legend box unobtrusive
    legend_background_color = :white,
    legend_foreground_color = :black,
)

# Exact bathymetry
plot!(p_leg, [NaN, NaN], [NaN, NaN],
    label = "Exact bathymetry",
    color = :black,
    lw = 1.5,
)

# 95% HPD
plot!(p_leg, [NaN, NaN], [NaN, NaN],
    ribbon=[.05,.05],
    label = "95% HPD",
    color = Plots.palette(:default)[1],
)

# Mean ± 95% CI
plot!(p_leg, [NaN, NaN], [NaN, NaN],
    ribbon=[.05,.05],
    label = "Mean ± 95% CI",
    color = Plots.palette(:default)[2],
)

# Sensor locations
scatter!(p_leg, [NaN], [NaN],
    label = "Sensor locations",
    marker = :star5,
    color = :black,
    ms = 6,
)

savefig(p_leg, "legend.pdf")

p_leg_sim = plot(
    legend = :bottom,
    legendcolumns = 2,
    legendfontsize = 10,

    framestyle = :none,
    grid = false,
    showaxis = false,

    left_margin = 0Plots.mm,
    right_margin = 0Plots.mm,
    top_margin = 0Plots.mm,
    bottom_margin = 0Plots.mm,

    size=(450,60),

    # Make the legend box unobtrusive
    legend_background_color = :white,
    legend_foreground_color = :black,
)

plot!(p_leg_sim, [NaN, NaN], [NaN, NaN],
    label = "Measurement",
    color = Plots.palette(:default)[1],
    linestyle = :dash,
    lw = 1.5,
)

plot!(p_leg_sim, [NaN, NaN], [NaN, NaN],
    label = "Simulation",
    color = Plots.palette(:default)[2],
    linestyle = :dot,
    lw = 1.5,
)

savefig(p_leg_sim, "legend_sim.pdf")