using Plots
plot_font = "Computer Modern"

# Define desired sizes in points (for PDF output)
desired_width_pt = 390# 1 point in Plots.jl units

pt2LaTex = 72/72.27

textwidth = desired_width_pt*pt2LaTex # 390 points for PDF
plot_height = Int(round(390 * 0.618 * pt2LaTex)) # Golden ratio for height

add_theme(:custom, PlotTheme(
    fontfamily = plot_font,
    linewidth = 1,
    markersize = 3,
    labelfontsize = 8,
    tickfontsize = 8,
    legendfontsize = 8,
    titlefontsize = 8,
    framestyle = :box,
    label = nothing,
    size = (textwidth, plot_height),  # 390pt width for PDF
    dpi=72
))