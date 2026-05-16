from plotnine import *

def custom_points(color='#FF4136', fill='#FF851B'):
     return geom_point(color=color, fill=fill, size=5, stroke=1.5)

def clean_theme(xrange=(-3, 10), yrange=(-3, 10), xlabel="", ylabel="", xintercept=0, yintercept=0):
    return [
    geom_hline(yintercept=yintercept, color='#666666', size=1),
    geom_vline(xintercept=xintercept, color='#666666', size=1),
    xlim(*xrange), 
    ylim(*yrange),        
    scale_x_continuous(expand=(0, 0), limits=xrange, breaks=None, labels=None),
    scale_y_continuous(expand=(0, 0), limits=yrange, breaks=None, labels=None),
    theme_minimal(),
    theme(
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        axis_text=element_blank(),
        axis_ticks=element_blank()
    ),
    labs(x=xlabel, y=ylabel)
]