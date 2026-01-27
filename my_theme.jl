using Plots
using Measures
plot_font = "Computer Modern"

textwidth = 319  # in pts for a standard LaTeX article document
plot_height = Int(round(textwidth * 0.618)) # Golden ratio for height

add_theme(:custom, PlotTheme(
    fontfamily = plot_font,
    linewidth = 2pt,
    labelfontsize = 12pt,
    tickfontsize = 12pt,
    legendfontsize = 12pt,
    titlefontsize = 12pt,
    framestyle = :box,
    label = nothing,
    size = (textwidth *pt, plot_height *pt),  # 319pt width
    dpi = 600
))