
import matplotlib
import matplotlib.pyplot as plt
import numpy
import scipy
import scipy.stats

_unique_colors = [
    'red',
    'blue',
    'magenta',
    'green',
    'orange',
    'cyan'
]

def get_series_extent(series,
        sigma_cutoff=4,
        use_robust=True):
    """
    Returns [min, max], where
    min = mu-sigma_cutoff*std,
    max = mu+sigma_cutoff*std.
    mu and sigma are mean and std dev values.
    use_robust parameter can be used to compute robustly.
    """
    # Determine mean and std dev values.
    if use_robust:
        mu = series.median()
        sigma = scipy.stats.median_abs_deviation(
            series, scale='normal', nan_policy='omit')
    else:
        mu = series.mean()
        sigma = series.std()

    # Compute min and max.
    min_val = mu - sigma_cutoff * sigma
    max_val = mu + sigma_cutoff * sigma
    return [min_val, max_val]

def get_group_extent(grouped,
        sigma_cutoff=4,
        use_robust=True):
    """
    Determines [min, max], where the min and max are taken
    by calling get_series_extent across all groups.
    """
    max_val = numpy.NINF
    min_val = numpy.Inf
    for name, group in grouped:
        extent = get_series_extent(group, sigma_cutoff, use_robust)
        min_val = min(extent[0], min_val)
        max_val = max(extent[1], max_val)
    return [min_val, max_val]

def quick_hist(grouped, num_bins=200, title=""):
    """
    Simple interface for plotting histograms of a groupby opject with labels.
    """
    # Set figure and figure properties.
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    alpha = 0.5
    ax.set_prop_cycle(color=_unique_colors)
    
    # Histogram sizes.
    extent = get_group_extent(grouped)
    hist_bins = numpy.linspace(extent[0], extent[1], num_bins)
    bin_centers = (hist_bins[1:] + hist_bins[:-1])/2.0

    # Plot histogram for each group.
    for name, group in grouped:
        vals = numpy.histogram(group, bins=hist_bins)
        ax.plot(bin_centers, vals[0], '-',
                label=name)
        ax.fill_between(bin_centers, 0, vals[0], alpha=alpha)

    # Show axis values.
    ax.set_title(title)
    ax.legend()
    ax.grid(which='both')

