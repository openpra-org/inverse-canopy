from .params import ModelParams


def subplot_pdfs(y_pred_pdf, y_target_pdf, title='',
                 event_names=['SDFR-0', 'SDFR-1', 'SDFR-2', 'SDFR-3', 'SDFR-4', 'SDFR-5', 'SDFR-6', 'SDFR-7', 'SDFR-8'],
                 save=False, alpha=0.2, size=(20, 5), xlim=[1e-10, 1e0], points=1000,
                 loc=['center left', 'center right']):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from matplotlib.ticker import LogLocator

    # Convert tensors to numpy arrays if they are not already
    y_pred_pdf = y_pred_pdf.numpy() if hasattr(y_pred_pdf, 'numpy') else y_pred_pdf
    y_target_pdf = y_target_pdf.numpy() if hasattr(y_target_pdf, 'numpy') else y_target_pdf

    # Set up the plot with two subplots
    fig, axs = plt.subplots(2, 1, figsize=size, sharex=True, constrained_layout=True)

    # Colors for each distribution
    # colors = plt.cm.tab10(np.linspace(0, 1, len(y_pred_pdf)))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']

    # Define a common grid to evaluate all densities
    x_grid = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), points, endpoint=False)

    # Function to plot distributions
    def plot_distributions(ax, y_pdf, title, loc_):
        for i, dist in enumerate(y_pdf):
            kde = gaussian_kde(dist, bw_method=1)
            log_dist = np.log(dist)
            mean, std = ModelParams(mus=np.mean(log_dist), sigmas=np.std(log_dist)).to_meanstd().spread()
            density = kde.evaluate(x_grid)
            if density.max() > 0:  # Check if density.max() is greater than zero
                # Normalize the peak of the density to 1 for comparison
                density /= density.max()
                ax.fill_between(x_grid, 0, density, color=colors[i], alpha=alpha)
                ax.plot(x_grid, density, label=f"{event_names[i]}: {mean:.2e} ± {std:.2e}", color=colors[i], alpha=1.0)
            else:
                ax.axvline(x=mean, linestyle='-', label=f"{event_names[i]}: {mean:.2e} ± {std:.2e}", color=colors[i],
                           alpha=1.0)
        if title != '':
            ax.set_title(title)
        ax.legend(framealpha=1, prop={'family': 'monospace'}, loc=loc_)

    # Plot distributions for y_pred_pdf and y_target_pdf
    plot_distributions(axs[0], y_pred_pdf, '', loc_=loc[0])
    plot_distributions(axs[1], y_target_pdf, '', loc_=loc[1])

    # Set labels and scales
    axs[1].set_xlabel('P(x)')
    axs[0].set_ylabel('Normalized Density - P(x)')
    axs[1].set_ylabel('Normalized Density - P(x)')
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')

    # set limits
    axs[0].set_xlim(xlim[0], xlim[1])
    axs[1].set_xlim(xlim[0], xlim[1])

    # Set major ticks every 10^-1
    major_locator = LogLocator(base=10.0, subs=[1.0], numticks=10)
    axs[0].xaxis.set_major_locator(major_locator)
    axs[1].xaxis.set_major_locator(major_locator)

    # Set minor ticks every 10^-2
    minor_locator = LogLocator(base=10.0, subs=np.linspace(2, 10, 9) * 0.1, numticks=100)
    axs[0].xaxis.set_minor_locator(minor_locator)
    axs[1].xaxis.set_minor_locator(minor_locator)

    # Optionally, improve the appearance of minor ticks
    axs[0].tick_params(which='minor', length=4, color='black', width=0.5)
    axs[1].tick_params(which='minor', length=4, color='black', width=0.5)

    # Adjust gridlines
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5, color='black', alpha=0.4, axis='x')
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='black', alpha=0.4, axis='x')

    if save:
        plt.savefig(f'{title}.png', format='png', dpi=300)  # dpi is more for compatibility here
        try:
            from google.colab import files
            files.download(f'{title}.png')
        except ImportError:
            print("File saved locally; automatic download not available outside Google Colab.")

    plt.show()
