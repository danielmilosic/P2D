def plot_timeseries(timeseries_data):
    import matplotlib.pyplot as plt
    import seaborn as sns

    timeseries_data = timeseries_data.dropna(axis=1, how='all')
    
    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(10, 12), sharex=True)

    # Panel 0: V and V_R
    if 'V' in timeseries_data.columns:
        sns.lineplot(data=timeseries_data, x=timeseries_data.index, y='V', ax=axes[0], color='black')
    else:
        axes[0].text(0.98, 0.7, "Missing: V", transform=axes[0].transAxes,
                     ha='right', va='center', color='black', fontsize=9)
    if 'V_R' in timeseries_data.columns:
        sns.lineplot(data=timeseries_data, x=timeseries_data.index, y='V_R', ax=axes[0], color='blue', alpha=0.5)
    else:
        axes[0].text(0.98, 0.5, "Missing: V_R", transform=axes[0].transAxes,
                     ha='right', va='center', color='black', fontsize=9)
    axes[0].set_ylabel('V $[km s^{-1}]$')

    # Panel 1: V_T and V_N
    if 'V_T' in timeseries_data.columns:
        sns.lineplot(data=timeseries_data, x=timeseries_data.index, y='V_T', ax=axes[1], label='V_T')
    else:
        axes[1].text(0.98, 0.7, "Missing: V_T", transform=axes[1].transAxes,
                     ha='right', va='center', color='black', fontsize=9)
    if 'V_N' in timeseries_data.columns:
        sns.lineplot(data=timeseries_data, x=timeseries_data.index, y='V_N', ax=axes[1], label='V_N')
    else:
        axes[1].text(0.98, 0.5, "Missing: V_N", transform=axes[1].transAxes,
                     ha='right', va='center', color='black', fontsize=9)
    axes[1].set_ylabel('V_TN $[km s^{-1}]$')

    # Panel 2: N and T (right y-axis)
    if 'N' in timeseries_data.columns:
        sns.lineplot(data=timeseries_data, x=timeseries_data.index, y='N', ax=axes[2], color='black')
    else:
        axes[2].text(0.98, 0.7, "Missing: N", transform=axes[2].transAxes,
                     ha='right', va='center', color='black', fontsize=9)
    axes[2].set_ylabel('N $[cm^{-3}]$')

    if 'T' in timeseries_data.columns:
        ax2 = axes[2].twinx()
        sns.lineplot(data=timeseries_data, x=timeseries_data.index, y='T', ax=ax2, color='tab:blue')
        ax2.set_ylabel('T $[K]$')
        ax2.spines['right'].set_color('tab:blue')
        ax2.yaxis.label.set_color('tab:blue')
        ax2.tick_params(axis='y', colors='tab:blue')
    else:
        axes[2].text(0.98, 0.3, "Missing: T", transform=axes[2].transAxes,
                     ha='right', va='center', color='tab:blue', fontsize=9)

    # Panel 3: P and Beta (right y-axis)
    if 'P' in timeseries_data.columns:
        sns.lineplot(data=timeseries_data, x=timeseries_data.index, y='P', ax=axes[3], color='black')
    else:
        axes[3].text(0.98, 0.7, "Missing: P", transform=axes[3].transAxes,
                     ha='right', va='center', color='black', fontsize=9)
    axes[3].set_ylabel('P $[nPa]$')

    if 'Beta' in timeseries_data.columns:
        ax3 = axes[3].twinx()
        sns.lineplot(data=timeseries_data, x=timeseries_data.index, y='Beta', ax=ax3, color='tab:blue')
        ax3.set_ylabel(r'$\beta$')
        ax3.set_yscale('log')
        ax3.spines['right'].set_color('tab:blue')
        ax3.yaxis.label.set_color('tab:blue')
        ax3.tick_params(axis='y', colors='tab:blue')
    else:
        axes[3].text(0.98, 0.3, "Missing: Beta", transform=axes[3].transAxes,
                     ha='right', va='center', color='tab:blue', fontsize=9)

    # Panel 4: B with polarity
    if 'B' in timeseries_data.columns and 'POL' in timeseries_data.columns:
        timeseries_data['polarity'] = ['+' if pol > 0 else '-' for pol in timeseries_data['POL']]
        colors = {'-': 'tab:blue', '+': 'tab:red'}
        sns.scatterplot(data=timeseries_data, x=timeseries_data.index, y='B',
                        hue='polarity', palette=colors, ax=axes[4], s=5, alpha=1)
        axes[4].get_legend().remove()
    else:
        if 'B' not in timeseries_data.columns:
            axes[4].text(0.98, 0.7, "Missing: B", transform=axes[4].transAxes,
                         ha='right', va='center', color='black', fontsize=9)
        if 'POL' not in timeseries_data.columns:
            axes[4].text(0.98, 0.5, "Missing: POL", transform=axes[4].transAxes,
                         ha='right', va='center', color='black', fontsize=9)
    axes[4].set_ylabel('B $[nT]$')

    # Panel 5: B_R, B_T, B_N
    for component, color, ypos in zip(['B_R', 'B_T', 'B_N'], ['red', 'green', 'blue'], [0.7, 0.5, 0.3]):
        if component in timeseries_data.columns:
            sns.lineplot(data=timeseries_data, x=timeseries_data.index, y=component, ax=axes[5], color=color, label=component)
        else:
            axes[5].text(0.98, ypos, f"Missing: {component}", transform=axes[5].transAxes,
                         ha='right', va='center', color=color, fontsize=9)
    axes[5].set_ylabel('B $[nT]$')

    # Panel 6: S_P and O7_O6_RATIO (right y-axis)
    if 'S_P' in timeseries_data.columns:
        sns.lineplot(data=timeseries_data, x=timeseries_data.index, y='S_P', ax=axes[6], color='black')
        axes[6].fill_between(timeseries_data.index, 2.69, 4, color='grey', alpha=0.7)
    else:
        axes[6].text(0.98, 0.7, "Missing: S_P", transform=axes[6].transAxes,
                     ha='right', va='center', color='black', fontsize=9)
    axes[6].set_ylabel('$S_p$ $[eV cm^{2}]$')

    if 'O7_O6_RATIO' in timeseries_data.columns:
        ax6 = axes[6].twinx()
        sns.scatterplot(data=timeseries_data, x=timeseries_data.index, y='O7_O6_RATIO', ax=ax6, s=5, color='tab:blue')
        sns.lineplot(data=timeseries_data, x=timeseries_data.index,
                     y=[0.145]*len(timeseries_data), ax=ax6, color='grey')
        ax6.set_ylim([0, 0.3])
        ax6.set_ylabel('$O^{7+}/O^{6+}$')
        ax6.spines['right'].set_color('tab:blue')
        ax6.yaxis.label.set_color('tab:blue')
        ax6.tick_params(axis='y', colors='tab:blue')
    else:
        axes[6].text(0.98, 0.3, "Missing: O7_O6_RATIO", transform=axes[6].transAxes,
                     ha='right', va='center', color='tab:blue', fontsize=9)

    # Panel 7: R and LAT (right y-axis)
    if 'R' in timeseries_data.columns:
        sns.lineplot(data=timeseries_data, x=timeseries_data.index, y='R', ax=axes[7], color='black')
    else:
        axes[7].text(0.98, 0.7, "Missing: R", transform=axes[7].transAxes,
                     ha='right', va='center', color='black', fontsize=9)
    axes[7].set_ylabel('r $[AU]$')

    if 'LAT' in timeseries_data.columns:
        ax7 = axes[7].twinx()
        sns.lineplot(data=timeseries_data, x=timeseries_data.index, y='LAT', ax=ax7, color='tab:blue')
        ax7.set_ylabel('LAT $[°]$')
        ax7.spines['right'].set_color('tab:blue')
        ax7.yaxis.label.set_color('tab:blue')
        ax7.tick_params(axis='y', colors='tab:blue')
    else:
        axes[7].text(0.98, 0.3, "Missing: LAT", transform=axes[7].transAxes,
                     ha='right', va='center', color='tab:blue', fontsize=9)

    plt.tight_layout(pad=1., w_pad=0.5, h_pad=.1)
