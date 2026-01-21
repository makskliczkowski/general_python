"""
Contains help strings for plotters.
"""

PLOTTER_HELP = {
            'overview': '''
╔══════════════════════════════════════════════════════════════════════════════╗
║                     PLOTTER - Scientific Plotting Utilities                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Topics (use Plotter.help('topic_name') for details):

'plot'    - Basic plotting: plot, scatter, semilogy, semilogx, loglog,
            errorbar, fill_between, histogram, contourf

'axis'    - Axis setup: set_ax_params, set_tickparams, setup_log_x,
            setup_log_y, set_smart_lim, unset_spines, unset_ticks, unset_all

'color'   - Colors: add_colorbar, get_colormap, discrete_colormap,
            Color cycles: colorsCycle, colorsCyclePlastic, etc.

'layout'  - Subplots: get_subplots, make_grid, GridBuilder, get_inset

'grid'    - Advanced grids: make_grid (full control), GridBuilder (nested)
            width_ratios, height_ratios, wspace, hspace, margins

'legend'  - Legends: set_legend (with style='publication'), set_legend_custom

'annotate'- Annotations: set_annotate, set_annotate_letter, set_arrow

'style'   - configure_style('nature'), configure_style('science'), 'prl', 'aps'

'save'    - Saving: save_fig(directory, filename, format='pdf', dpi=300)

──────────────────────────────────────────────────────────────────────────────
Quick Example:

    fig, axes = Plotter.get_subplots(1, 2, sizex=8, sizey=3)
    Plotter.plot(axes[0], x, y, label='Data', color='C0')
    Plotter.semilogy(axes[1], x, y_log, color='C1')
    Plotter.set_ax_params(axes[0], xlabel='x', ylabel='y', title='Linear')
    Plotter.set_legend(axes[0], style='publication')
    Plotter.save_fig('.', 'my_figure', format='pdf')
──────────────────────────────────────────────────────────────────────────────
''',
            
            'plot': '''
================
PLOTTING METHODS
================

Basic Plots:
    Plotter.plot(ax, x, y, ls='-', lw=2, color='black', label=None, marker=None)
    Plotter.scatter(ax, x, y, s=10, c='blue', marker='o', alpha=1.0, label=None)

Logarithmic Scales:
    Plotter.semilogy(ax, x, y, **kwargs)    # log y-axis
    Plotter.semilogx(ax, x, y, **kwargs)    # log x-axis  
    Plotter.loglog(ax, x, y, **kwargs)      # log both axes

Error Bars:
    Plotter.errorbar(ax, x, y, yerr=None, xerr=None, fmt='o', capsize=2, **kwargs)

Histogram:
    Plotter.histogram(ax, data, bins=50, density=True, alpha=0.7, **kwargs)

Filled Regions:
    Plotter.fill_between(ax, x, y1, y2, color='blue', alpha=0.5)

Reference Lines:
    Plotter.hline(ax, val, ls='--', lw=2, color='black', label=None)
    Plotter.vline(ax, val, ls='--', lw=2, color='black', label=None)

Contour:
    cs = Plotter.contourf(ax, x, y, z, levels=20, cmap='viridis')
''',
            
            'axis': '''
==================
AXIS CONFIGURATION
==================

Comprehensive Setup:
    Plotter.set_ax_params(
        ax,
        xlabel='x', ylabel='y',          # Axis labels
        xlim=(0, 10), ylim=(0, 1),        # Axis limits
        xscale='linear', yscale='log',   # Scale ('linear', 'log')
        fontsize=12,                      # Label font size
        title='My Plot',                  # Plot title
    )

Log-Scale Setup (with proper tick formatting):
    Plotter.setup_log_y(ax, ylims=(1e-6, 1e0), decade_step=2)
    Plotter.setup_log_x(ax, xlims=(1e-3, 1e3), decade_step=2)

Tick Parameters:
    Plotter.set_tickparams(ax, labelsize=10, maj_tick_l=6, min_tick_l=3)

Smart Limits (auto-compute from data):
    Plotter.set_smart_lim(ax, which='y', data=my_data, margin_p=0.1)

──────────────────────────────────────────────────────────────────────────────
REMOVING SPINES & TICKS (Publication Style)
──────────────────────────────────────────────────────────────────────────────

Remove Spines:
    # Remove specific spines (True = REMOVE, False = KEEP)
    Plotter.unset_spines(ax, top=True, right=True, bottom=False, left=False)
    
    # Nature-style (only left and bottom spines)
    Plotter.unset_spines(ax, top=True, right=True)
    
    # Remove all spines (frameless plot)
    Plotter.unset_spines(ax, top=True, right=True, bottom=True, left=True)

Remove Tick Labels (keep spines):
    # Hide x-tick labels (useful for stacked plots)
    Plotter.unset_ticks(ax, xticks=True, yticks=False)
    
    # Hide all tick labels
    Plotter.unset_ticks(ax, xticks=True, yticks=True)
    
    # Also remove axis labels
    Plotter.unset_ticks(ax, xticks=True, xlabel=True)

Scientific Notation:
    set_formatter(ax, formatter_type='sci', fmt='%1.1e', axis='y')

──────────────────────────────────────────────────────────────────────────────
QUICK REFERENCE - Common Axis Operations
──────────────────────────────────────────────────────────────────────────────

# Shared x-axis in stacked plots (hide middle ticks)
for ax in axes[:-1]:
    Plotter.unset_ticks(ax, xticks=True, xlabel=True)

# Clean Nature-style axes
for ax in axes:
    Plotter.unset_spines(ax, top=True, right=True)

# Log scale with nice decade ticks
Plotter.setup_log_y(ax, ylims=(1e-6, 1e0), decade_step=1)
''',
            
            'color': r'''
==================
COLORS & COLORBARS
==================

Add Colorbar:
    cbar, cax = Plotter.add_colorbar(
        fig, pos=[0.92, 0.15, 0.02, 0.7],  # [left, bottom, width, height]
        mappable=data,                      # or ScalarMappable
        cmap='viridis',
        scale='log',                        # 'linear', 'log', 'symlog'
        label=r'$|\psi|^2$',
        orientation='vertical',
    )

Get Colormap Function:
    getcolor, cmap, norm = Plotter.get_colormap(values, cmap='coolwarm')
    color = getcolor(value)  # Returns RGBA for value

Discrete Colormap:
    cmap = Plotter.discrete_colormap(N=5, base_cmap='viridis')

Available Color Cycles (use next(cycle) to get colors):
    colorsCycle         - Tableau colors (default)
    colorsCyclePlastic  - Colorblind-safe palette
    colorsCycleBright   - CSS4 colors
    colorsCyclePastel   - XKCD colors

Reset Cycles:
    reset_color_cycles()  # Reset all
    reset_color_cycles('Plastic')  # Reset specific
''',
            
            'layout': r'''
=================
LAYOUT & SUBPLOTS
=================

Simple Subplots (recommended for quick plots):
    fig, axes = Plotter.get_subplots(
        nrows=2, ncols=3,
        sizex=10, sizey=6,                # Total figure size in inches
        panel_labels=True,                # Add (a), (b), (c) labels
        despine=True,                     # Remove top/right spines
        constrained_layout=True,          # Auto-adjust spacing
    )
    # axes is always a flat list: [ax0, ax1, ax2, ...]

──────────────────────────────────────────────────────────────────────────────
ADVANCED GRIDS - Full Control Over Layout
──────────────────────────────────────────────────────────────────────────────

Plotter.make_grid() - Static Method for Complex Layouts:
    fig, axes = Plotter.make_grid(
        nrows=2, ncols=3,
        figsize=(10, 6),
        
        # Column/Row Sizing
        width_ratios=[1, 2, 1],           # Column width proportions
        height_ratios=[1, 3],              # Row height proportions
        
        # Spacing Control
        wspace=0.3,                        # Horizontal gap (0-1)
        hspace=0.2,                        # Vertical gap (0-1)
        left=0.1, right=0.95,              # Figure margins
        top=0.95, bottom=0.1,
        
        # Per-Axis Options
        sharex='col',                      # 'row', 'col', 'all', False
        sharey='row',
        
        # Publication Options
        panel_labels=True,                 # Add (a), (b), (c)
        panel_label_style='parenthesis',   # 'parenthesis', 'plain', 'bold'
        despine=True,
    )

──────────────────────────────────────────────────────────────────────────────
GridBuilder - Class for Maximum Flexibility
──────────────────────────────────────────────────────────────────────────────

For nested/complex layouts, use the GridBuilder class:

    builder = Plotter.GridBuilder(figsize=(12, 8))
    
    # Add a full-width row at top
    builder.add_row(ncols=1, height_ratio=1)
    
    # Add 3-column row with custom width ratios
    builder.add_row(ncols=3, height_ratio=2, width_ratios=[2, 1, 1])
    
    # Add 2-column row
    builder.add_row(ncols=2, height_ratio=1.5)
    
    # Build the figure
    fig, axes = builder.build()
    # axes is a list of lists: [[ax00], [ax10, ax11, ax12], [ax20, ax21]]

GridBuilder supports:
    .add_row(ncols, height_ratio=1, width_ratios=None)  # Add horizontal row
    .add_column(nrows, width_ratio=1, height_ratios=None)  # Add vertical column
    .add_subplot(rows=1, cols=1)  # Add to current position (call multiple times)
    .build(wspace=0.2, hspace=0.2, **margins)  # Finalize

──────────────────────────────────────────────────────────────────────────────
COMMON LAYOUT RECIPES
──────────────────────────────────────────────────────────────────────────────

# Recipe 1: Two-column figure with shared x-axis
fig, axes = Plotter.make_grid(2, 1, figsize=(4, 6), height_ratios=[2, 1], 
                               sharex='col', hspace=0.05)

# Recipe 2: Main plot with colorbar column
fig, axes = Plotter.make_grid(1, 2, figsize=(6, 4), width_ratios=[20, 1])
ax_main, ax_cbar = axes

# Recipe 3: 2x2 grid with panel labels for publication
fig, axes = Plotter.make_grid(2, 2, figsize=(8, 8), panel_labels=True, despine=True)

# Recipe 4: Asymmetric layout using GridBuilder
builder = Plotter.GridBuilder(figsize=(12, 6))
builder.add_row(ncols=1, height_ratio=1)     # Wide plot at top
builder.add_row(ncols=3, height_ratio=2)     # 3 panels below
fig, axes = builder.build(hspace=0.3)

──────────────────────────────────────────────────────────────────────────────
INSETS & OVERLAYS
──────────────────────────────────────────────────────────────────────────────

Inset Axes:
    inset = Plotter.get_inset(
        ax,
        position=[0.6, 0.6, 0.35, 0.35],  # [x, y, width, height] relative to ax
        add_box=True,                      # Semi-transparent background
    )

Twin Axes (secondary y-axis):
    ax2 = Plotter.twin_axis(ax, which='y', label='Secondary Y', color='C1')
    ax2.plot(x, secondary_data, color='C1')

Unify Axis Limits Across Panels:
    Plotter.unify_limits(axes, which='y')  # Same y-limits for all
''',
            
            'legend': r'''
=======
LEGENDS
=======

Publication-Style Legend:
    Plotter.set_legend(
        ax,
        style='publication',    # 'minimal', 'boxed', 'publication'
        loc='best',
        ncol=1,
        frameon=True,
        fontsize=10,
    )

Custom Legend (filter by condition):
    Plotter.set_legend_custom(
        ax,
        conditions=[lambda lbl: 'exp' in lbl],  # Only labels containing 'exp'
    )

Available Styles:
    'minimal'     - No frame, small font
    'boxed'       - Semi-transparent frame
    'publication' - Solid frame, Nature-style formatting
''',
            
            'save': r'''
==============
SAVING FIGURES
==============

Save Figure:
    Plotter.save_fig(
        directory='./figures',
        filename='my_plot',
        format='pdf',           # 'pdf', 'png', 'svg', 'eps', 'tiff'
        dpi=300,                # For rasterized formats
        adjust=True,            # Tight bounding box
    )

Recommended Formats:
    - PDF/EPS: Vector format for journals (scalable, small file)
    - PNG: Rasterized for presentations (transparent background)
    - TIFF: High-resolution for print (300+ dpi)
    - SVG: Web/interactive (editable in Inkscape/Illustrator)

Tip: Use constrained_layout=True in get_subplots() to avoid clipping.
''',
            
            'grid': r'''
=====================
ADVANCED GRID CONTROL
=====================

For complex layouts with precise control over spacing, sizing, and alignment.

──────────────────────────────────────────────────────────────────────────────
Plotter.make_grid() - Recommended for Most Cases
──────────────────────────────────────────────────────────────────────────────

    fig, axes = Plotter.make_grid(
        nrows=2, ncols=3,
        figsize=(10, 6),
        
        # Size Ratios (list matching number of rows/cols)
        width_ratios=[1, 2, 1],     # Columns: narrow, wide, narrow
        height_ratios=[1, 3],        # Rows: short header, tall main
        
        # Spacing (0-1, fraction of average subplot size)
        wspace=0.3,                  # Horizontal gap between columns
        hspace=0.2,                  # Vertical gap between rows
        
        # Figure Margins (0-1, fraction of figure size)
        left=0.1, right=0.95,
        top=0.95, bottom=0.1,
        
        # Axis Sharing
        sharex='col',                # 'row', 'col', 'all', or False
        sharey='row',
    )

──────────────────────────────────────────────────────────────────────────────
Plotter.GridBuilder - For Nested/Complex Layouts
──────────────────────────────────────────────────────────────────────────────

    builder = Plotter.GridBuilder(figsize=(12, 8))
    
    # Add rows with different column configurations
    builder.add_row(ncols=1, height_ratio=1)           # Single wide panel
    builder.add_row(ncols=3, height_ratio=2,           # 3 columns
                    width_ratios=[2, 1, 1])            # with custom widths
    builder.add_row(ncols=2, height_ratio=1.5)         # 2 columns
    
    fig, axes = builder.build(wspace=0.2, hspace=0.3)
    # axes = [[ax00], [ax10, ax11, ax12], [ax20, ax21]]

──────────────────────────────────────────────────────────────────────────────
Example Recipes
──────────────────────────────────────────────────────────────────────────────

# Stacked panels with shared x-axis (no gap)
fig, axes = Plotter.make_grid(3, 1, figsize=(6, 8), hspace=0.05, sharex='col')
for ax in axes[:-1]:
    Plotter.unset_ticks(ax, xticks=True, xlabel=True)

# Main plot + colorbar
fig, axes = Plotter.make_grid(1, 2, figsize=(7, 5), width_ratios=[20, 1])

# 2x2 for publication with panels (a)-(d)
fig, axes = Plotter.make_grid(2, 2, figsize=(8, 8), panel_labels=True)
''',
            
            'style': r'''
==================
PUBLICATION STYLES
==================

Configure global Matplotlib style for different journals.

    configure_style('nature')    # Nature, Science, Cell
    configure_style('science')   # Science-family journals
    configure_style('prl')       # Physical Review Letters
    configure_style('aps')       # APS journals (PRB, PRX, etc.)
    configure_style('default')   # Matplotlib defaults

What it Sets:
    - Font family (STIX/serif for publication)
    - Font sizes (labels, ticks, legend)
    - Line widths
    - Figure DPI
    - Axes styling
    - Color cycle

Example:
    from QES.general_python.common.plot import configure_style, Plotter
    
    configure_style('nature')  # Apply once at start
    
    fig, ax = Plotter.get_subplots(1, 1)
    Plotter.plot(ax, x, y)
    Plotter.save_fig('.', 'figure1', format='pdf')

Custom Adjustments:
    After configure_style(), you can fine-tune with:
    
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 0.8
''',
}

# ──────────────────────────────────────────────────────────────────────────────
#! EOF
# ──────────────────────────────────────────────────────────────────────────────