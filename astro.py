#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.timeseries import LombScargle
from astropy.io import fits
from astroquery.gaia import Gaia
from astroquery.mast import Observations
import requests
from scipy.optimize import curve_fit
import lightkurve as lk
import warnings

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

class RRdAnalyzer:
    def __init__(self, star_id, catalog='ogle'):
        self.star_id = star_id
        self.catalog = catalog.lower()
        self.time = None
        self.mag = None
        self.mag_err = None
        self.periods = {}
        self.fourier_params = {}

    def fetch_ogle_data(self):
        base_url = "http://ogledb.astrouw.edu.pl/~ogle/photdb/"
        field = self.star_id.split('-')[1]  # BLG, LMC, SMC
        url = f"{base_url}query.php?db=OGLE-IV&starcat=RRLyr&field={field}&starid={self.star_id}"
        try:
            response = requests.get(url, timeout=30)
            data = np.genfromtxt(response.text.splitlines(), skip_header=1)
            self.time = data[:, 0]
            self.mag = data[:, 1]
            self.mag_err = data[:, 2] if data.shape[1] > 2 else np.ones_like(self.mag) * 0.01
            return True
        except:
            print(f"OGLE fetch failed for {self.star_id}")
            return False

    def fetch_kepler_data(self):
        try:
            search_result = lk.search_lightcurve(self.star_id, mission='K2', author='K2')
            if len(search_result) == 0:
                search_result = lk.search_lightcurve(self.star_id, mission='Kepler', author='Kepler')
            search_result = search_result[search_result.author == 'K2']
            if len(search_result) == 0:
                search_result = lk.search_lightcurve(self.star_id, mission='K2')
            if len(search_result) > 0:
                lc = search_result[0].download()
                self.time = lc.time.value
                self.mag = lc.flux.value
                self.mag = self.mag / np.nanmedian(self.mag)
                self.mag_err = np.ones_like(self.mag) * 0.01
                mask = np.isfinite(self.time) & np.isfinite(self.mag)
                self.time = self.time[mask]
                self.mag = self.mag[mask]
                self.mag_err = self.mag_err[mask]
                return True
            return False
        except Exception as e:
            print(f"Kepler fetch failed: {e}")
            return False

    def fetch_gaia_data(self):
        try:
            query = f"""
            SELECT source_id, ra, dec, phot_g_mean_mag, parallax, pmra, pmdec
            FROM gaiadr3.gaia_source
            WHERE CONTAINS(POINT('ICRS', ra, dec), 
                          CIRCLE('ICRS', {self.ra}, {self.dec}, 0.0014))=1
            """
            job = Gaia.launch_job_async(query)
            gaia_table = job.get_results()
            return gaia_table[0] if len(gaia_table) > 0 else None
        except:
            return None

    def run_multiharmonic_ls(self, n_harmonics=5, freq_range=(0.1, 10)):
        frequency = np.linspace(freq_range[0], freq_range[1], 10000)
        ls = LombScargle(self.time, self.mag, self.mag_err)
        power = ls.power(frequency)
        best_freq = frequency[np.argmax(power)]
        self.periods['f0'] = 1 / best_freq
        residuals = self.mag.copy()
        for i in range(n_harmonics):
            ls_res = LombScargle(self.time, residuals, self.mag_err)
            power_res = ls_res.power(frequency)
            harm_freq = frequency[np.argmax(power_res)]
            self.periods[f'f{i}'] = 1 / harm_freq
            model = ls_res.model(self.time, harm_freq)
            residuals -= model

        return frequency, power, residuals

    def run_aov_periodogram(self, n_bins=10):
        periods = np.linspace(0.1, 2.0, 5000)
        aov_power = np.zeros_like(periods)
        for i, period in enumerate(periods):
            phase = (self.time % period) / period
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(phase, bins)
            bin_means = []
            for j in range(1, n_bins + 1):
                mask = bin_indices == j
                if np.sum(mask) > 0:
                    bin_means.append(np.mean(self.mag[mask]))
            if len(bin_means) > 0:
                overall_mean = np.mean(self.mag)
                s_between = np.var(bin_means) * len(bin_means)
                s_total = np.var(self.mag)
                aov_power[i] = s_between / s_total if s_total > 0 else 0
        best_period = periods[np.argmax(aov_power)]
        self.periods['aov_p0'] = best_period
        return periods, aov_power

    def fourier_decomposition(self, period, n_orders=8):
        phase = (self.time % period) / period
        sorted_idx = np.argsort(phase)
        phase_sorted = phase[sorted_idx]
        mag_sorted = self.mag[sorted_idx]
        def fourier_series(x, A0, *params):
            result = A0 * np.ones_like(x)
            n = len(params) // 2
            for k in range(n):
                A_k = params[2 * k]
                phi_k = params[2 * k + 1]
                result += A_k * np.cos(2 * np.pi * (k + 1) * x - phi_k)
            return result
        p0 = [np.mean(mag_sorted)]
        for k in range(n_orders):
            p0.extend([0.1, 0])
        try:
            popt, _ = curve_fit(fourier_series, phase_sorted, mag_sorted, p0=p0, maxfev=10000)
            A0 = popt[0]
            amplitudes = popt[1::2]
            phases = popt[2::2]
            phi21 = phases[1] - 2 * phases[0]
            phi31 = phases[2] - 3 * phases[0]
            R21 = amplitudes[1] / amplitudes[0] if amplitudes[0] != 0 else 0
            R31 = amplitudes[2] / amplitudes[0] if amplitudes[0] != 0 else 0
            self.fourier_params = {
                'A0': A0,
                'amplitudes': amplitudes,
                'phases': phases,
                'R21': R21,
                'R31': R31,
                'phi21': phi21,
                'phi31': phi31
            }
            phase_model = np.linspace(0, 1, 500)
            mag_model = fourier_series(phase_model, *popt)
            return phase_sorted, mag_sorted, phase_model, mag_model
        except:
            print("Fourier fit failed")
            return phase_sorted, mag_sorted, None, None

    def petersen_diagram(self, P0, P1):
        ratio = P1 / P0
        log_P0 = np.log10(P0)
        return log_P0, ratio

    def plot_results(self, figsize=(16, 12)):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.errorbar(self.time, self.mag, yerr=self.mag_err, fmt='k.',
                     markersize=2, alpha=0.5, elinewidth=0.5)
        ax1.set_xlabel('Time [days]')
        ax1.set_ylabel('Magnitude')
        ax1.set_title(f'{self.star_id} - Raw Light Curve', fontweight='bold')
        ax1.invert_yaxis()
        freq, power, residuals = self.run_multiharmonic_ls()
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(1 / freq, power, 'b-', linewidth=0.8)
        ax2.axvline(self.periods['f0'], color='r', linestyle='--',
                    label=f"P0={self.periods['f0']:.5f} d")
        ax2.set_xlabel('Period [days]')
        ax2.set_ylabel('LS Power')
        ax2.set_title('Lomb-Scargle Periodogram', fontweight='bold')
        ax2.legend()
        ax2.set_xlim(0.2, 1.0)
        periods_aov, power_aov = self.run_aov_periodogram()
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(periods_aov, power_aov, 'g-', linewidth=0.8)
        ax3.axvline(self.periods['aov_p0'], color='r', linestyle='--',
                    label=f"P0={self.periods['aov_p0']:.5f} d")
        ax3.set_xlabel('Period [days]')
        ax3.set_ylabel('AoV Statistic')
        ax3.set_title('Analysis of Variance', fontweight='bold')
        ax3.legend()
        if 'f0' in self.periods:
            phase_data, mag_data, phase_model, mag_model = \
                self.fourier_decomposition(self.periods['f0'])
            ax4 = fig.add_subplot(gs[1, 2])
            ax4.plot(phase_data, mag_data, 'k.', markersize=3, alpha=0.6, label='Data')
            if mag_model is not None:
                ax4.plot(phase_model, mag_model, 'r-', linewidth=2, label='Fourier fit')
            ax4.set_xlabel('Phase')
            ax4.set_ylabel('Magnitude')
            ax4.set_title('Phase-Folded LC (Fund.)', fontweight='bold')
            ax4.invert_yaxis()
            ax4.legend()
        ax5 = fig.add_subplot(gs[2, 0])
        if self.fourier_params:
            orders = np.arange(1, len(self.fourier_params['amplitudes']) + 1)
            ax5.bar(orders, self.fourier_params['amplitudes'], color='steelblue', alpha=0.7)
            ax5.set_xlabel('Fourier Order')
            ax5.set_ylabel('Amplitude')
            ax5.set_title('Fourier Amplitudes', fontweight='bold')
        ax6 = fig.add_subplot(gs[2, 1])
        if self.fourier_params:
            text_str = f"R21 = {self.fourier_params['R21']:.3f}\n"
            text_str += f"R31 = {self.fourier_params['R31']:.3f}\n"
            text_str += f"phi21 = {self.fourier_params['phi21']:.3f}\n"
            text_str += f"phi31 = {self.fourier_params['phi31']:.3f}"
            ax6.text(0.1, 0.5, text_str, fontsize=14, family='monospace',
                     verticalalignment='center', transform=ax6.transAxes)
            ax6.axis('off')
            ax6.set_title('Fourier Parameters', fontweight='bold')
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.plot(self.time, residuals, 'k.', markersize=2, alpha=0.5)
        ax7.set_xlabel('Time [days]')
        ax7.set_ylabel('Residual Mag')
        ax7.set_title('Residuals (Multi-harmonic)', fontweight='bold')
        ax7.axhline(0, color='r', linestyle='--', linewidth=1)
        plt.suptitle(f'RR Lyrae RRd Analysis: {self.star_id}',
                     fontsize=16, fontweight='bold', y=0.995)
        return fig


def analyze_star(star_id, catalog=None, ra=None, dec=None, save_fig=True):
    if catalog is None:
        if star_id.startswith('OGLE'):
            catalog = 'ogle'
        elif star_id.startswith('EPIC'):
            catalog = 'k2'
        elif star_id.startswith('KIC'):
            catalog = 'kepler'
        else:
            print(f"Cannot detect catalog for {star_id}")
            return None
    analyzer = RRdAnalyzer(star_id, catalog)
    print(f"Fetching data for {star_id}...")
    if catalog == 'ogle':
        success = analyzer.fetch_ogle_data()
    elif catalog in ['kepler', 'k2']:
        success = analyzer.fetch_kepler_data()
    else:
        print(f"Unknown catalog: {catalog}")
        return None
    if not success:
        print("Data fetch failed")
        return None
    if ra is not None and dec is not None:
        analyzer.ra = ra
        analyzer.dec = dec
        gaia_data = analyzer.fetch_gaia_data()
        if gaia_data:
            print(f"Gaia parallax: {gaia_data['parallax']:.3f} mas")
    print("Running analysis...")
    fig = analyzer.plot_results()
    if save_fig:
        filename = f"{star_id.replace(' ', '_')}_analysis.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
    #plt.show()
    return analyzer


def plot_petersen_diagram(analyzers, figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    log_P0_list = []
    ratio_list = []
    for analyzer in analyzers:
        if 'f0' in analyzer.periods and 'f1' in analyzer.periods:
            log_P0, ratio = analyzer.petersen_diagram(
                analyzer.periods['f0'], analyzer.periods['f1']
            )
            log_P0_list.append(log_P0)
            ratio_list.append(ratio)
    ax.scatter(log_P0_list, ratio_list, s=100, alpha=0.7,
               c='steelblue', edgecolors='black', linewidth=1.5)
    ax.set_xlabel('log P0 [days]', fontsize=12)
    ax.set_ylabel('P1/P0', fontsize=12)
    ax.set_title('Petersen Diagram - RR Lyrae RRd', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('petersen_diagram.png', dpi=300, bbox_inches='tight')
    #plt.show()
    return fig


if __name__ == "__main__":
    #star_id = "OGLE-BLG-RRLYR-12345"  # Replace with actual star ID
    # Single star analysis
    #analyzer = analyze_star(star_id, catalog='ogle')

    # For multiple stars and Petersen diagram:
    star_ids = ["EPIC216764000", "EPIC235839761", "EPIC251248823", "EPIC205209951", "EPIC60018653", "EPIC210282472", "EPIC211117230", "EPIC201585823", "EPIC206088888", "EPIC212640000", "EPIC249451861", "EPIC248926537", "EPIC206032309", "EPIC210282473"]
    analyzers = [analyze_star(sid, save_fig=True) for sid in star_ids]
    plot_petersen_diagram([a for a in analyzers if a is not None])

    print("\nAnalysis complete!")