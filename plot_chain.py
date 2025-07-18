import numpy as np
import matplotlib.pyplot as plt
import math
import corner
import dynesty
import glob
import os
from enterprise.signals import gp_signals
from enterprise_extensions import model_utils, blocks
from enterprise.signals import signal_base
from enterprise.pulsar import Pulsar
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
import enterprise.signals.parameter as parameter
from enterprise.signals import white_signals
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
from enterprise_extensions.blocks import white_noise_block, red_noise_block, dm_noise_block

parfiles = sorted(glob.glob('epta_sim_1/*.par'))
timfiles_new = sorted(glob.glob('epta_sim_1/*_1440.tim'))
psrs = []
ephemeris = None

for p, t in zip(parfiles, timfiles_new):
    psr = Pulsar(p, t, ephem=ephemeris)
    psrs.append(psr)

# find the maximum time span to set GW frequency sampling
Tspan = model_utils.get_tspan(psrs)

# Here we build the signal model
# First we add the timing model
s = gp_signals.TimingModel()

# Then we add the white noise
# We use different white noise parameters for every backend/receiver combination
# The white noise parameters are held constant
efac = parameter.Uniform(0.1, 5.0)
s += white_signals.MeasurementNoise(efac=efac)

s += red_noise_block(components=30) 
#s += dm_noise_block(components=30)

# Finally, we add the common red noise, which is modeled as a Fourier series with 30 frequency components
# The common red noise has a power-law PSD with spectral index of 4.33
s += blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan, components=30, name='gw_crn', orf = None)

# We set up the PTA object using the signal we defined above and the pulsars
pta = signal_base.PTA([s(p) for p in psrs])
print(pta.params)

chain = np.genfromtxt("report_sim_1/chain_1.txt")
burn = int(0.39 * chain.shape[0])
chain = chain[burn:, :]
print(chain[burn:,-6:-4].shape)
truths = [13/3, np.log10(3e-14)]#задаются эталонные значения параметров для сравнения с результатами MCMC.
corner.corner(chain[burn:,-6:-4], 30, truths=truths, labels=["gamma_gw", "Amp_gw"]);
plt.savefig("poster_rn.png", dpi=300)
#plt.show()
plt.clf()
#chain = np.genfromtxt("report_sim/chain_1.txt")
#burn = int(0.5 * chain.shape[0])
#chain = chain[burn:, :]

orf = ['hd']
pt = opt_stat.OptimalStatistic(psrs, pta=pta, orf="hd")
out = pt.compute_noise_marginalized_os(chain=chain, N=100)

snr_mean = np.mean(out[4])
plt.hist(out[4], bins=30, color='lightblue', edgecolor='black')
plt.axvline(snr_mean, color='blue', linestyle='--', linewidth=2, label=f'Среднее SNR = {snr_mean:.2f}')
#plt.hist(out[4])
plt.xlabel("SNR")
#plt.show()
plt.savefig("snr_rn_mean.png", dpi=300)
print('Ready')