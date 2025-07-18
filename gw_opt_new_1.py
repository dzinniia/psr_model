import os
import numpy as np
import matplotlib.pyplot as plt
import libstempo as T
import libstempo.plot as LP, libstempo.toasim as LT
import glob
import math
import json
from enterprise.signals import gp_signals
from enterprise_extensions import model_utils, blocks
import dynesty
from enterprise.signals import signal_base
from enterprise.pulsar import Pulsar
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
import enterprise.signals.parameter as parameter
from enterprise.signals import white_signals
from enterprise_extensions.blocks import white_noise_block, red_noise_block, dm_noise_block
from enterprise_extensions import sampler
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
import corner

parfiles = sorted(glob.glob('epta_init-20250505T112517Z-1-001/*.par'))
Npsr = len(parfiles)
datadir = "epta_sim_3/"
outdir = "report_sim_3/"

psrs = []
Amp = 3e-14
gamma = 13./3.
toas = np.arange(57000,57000+10*365.25,28.)
all_toas = []
all_freqs = []
freqs = [700, 1440, 2100]

for f in freqs:
    all_toas.extend(toas)
    all_freqs.extend([f] * len(toas))

all_toas = np.array(all_toas)
all_freqs = np.array(all_freqs)
log10A_dm = [-18.93, -10.51, -10.65, -10.87, -10.88, -10.39, -10.75, -18.93, -10.39, -10.59, -10.79, -10.49, -18.93, -10.75, -10.01, -9.66, -10.12, -9.96, -10.79, -10.78, -10.64, -10.91, -11.02, -10.7, -18.93]
gamma_dm = [3.1, 1.34, 2.69, 3.89, 1.74, 0.14, 2.46, 3.1, 1.99, 0.48, 1.59, 2.22, 3.1, 0.26, 2.13, 1.68, 0.78, 2.07, 2.88, 1.31, 2.98, 3.06, 3.49, 2.07, 3.1]

log10A_red = [-14.39, -14.99, -20, -12.76, -13.03, -13.8, -20, -13.26, -14.05, -20, -14.19, -20, -12.93, -14.12, -20, -20, -20, -20, -20, -14.89, -20, -20, -20, -20, -20]
gamma_red = [5.49, 5.34, 3.1, 1.06, 1.21, 3.01, 3.1, 2.21, 2.86, 3.1, 3.28, 3.1, 2.14, 3.45, 3.1, 3.1, 3.1, 3.1, 3.1, 4.77, 3.1, 3.1, 3.1, 3.1, 3.1]

    

for ii in range(Npsr):
    psr = LT.fakepulsar(parfile=parfiles[ii], obstimes=all_toas, freq=all_freqs, toaerr=0.1)
    LT.make_ideal(psr)
    LT.add_efac(psr, efac=1.0)
   
    LT.add_dm(psr, 10**log10A_dm[ii], gamma_dm[ii], components=30)
    LT.add_rednoise(psr, 10**log10A_red[ii], gamma_red[ii], components=30)

    psrs.append(psr)

LT.createGWB(psrs, Amp=Amp, gam=gamma, seed=5783)

for Psr in psrs:
    Psr.savepar(datadir + Psr.name + '.par')
    Psr.savetim(datadir + Psr.name + '.tim')
    T.purgetim(datadir + Psr.name + '.tim')

os.system("rm -f " + datadir + "*_1440.tim")

parfiles = sorted(glob.glob(datadir+'*.par'))
timfiles = sorted(glob.glob(datadir+'*.tim'))

for i in timfiles:
    os.system("awk '/FORMAT/' " + i + " >> " + i[:-4] + "_1440.tim")
    os.system("awk '/1440.00/' " + i + " >> " + i[:-4] + "_1440.tim")
timfiles_new = sorted(glob.glob(datadir+'*_1440.tim'))
psrs = []
ephemeris = None

for p, t in zip(parfiles, timfiles_new):
    psr = Pulsar(p, t, ephem=ephemeris)
    psrs.append(psr)



#freq = np.fft.fftfreq(len(psrs[1].residuals), 28)
#for jj in psrs[0:3]:
#    plt.plot(freq[:int(len(np.fft.fft(jj.residuals))/2)], np.abs(np.fft.fft(jj.residuals))[:int(len(np.fft.fft(jj.residuals))/2)])
#    plt.ylim(1e-7, 1e-3)
#    plt.xscale("log")
#    plt.yscale("log")
#plt.xlabel("Frequency, 1/day")
#plt.ylabel("Power")
#plt.savefig("spectra_red.png", dpi=300)

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
#s +=dm_noise_block(components=30)

# Finally, we add the common red noise, which is modeled as a Fourier series with 30 frequency components
# The common red noise has a power-law PSD with spectral index of 4.33
s += blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan, components=30, name='gw_crn', orf = None)

# We set up the PTA object using the signal we defined above and the pulsars
pta = signal_base.PTA([s(p) for p in psrs])
print(pta.params)
#%% Nested Sampling with dynesty

def TransformPrior(theta, verbose=False):
	if verbose:
		print('\n', len(theta), '\n')
	ph_pars = np.zeros(len(theta))
	idx_offset = 0
	for i, nm in enumerate(pta.params):
		if nm.size and nm.size>1:
			for j in range(nm.size):
				ph_pars[i+idx_offset] = (pta.params[i].prior.func_kwargs['pmax'] - pta.params[i].prior.func_kwargs['pmin']) * theta[i+idx_offset] + pta.params[i].prior.func_kwargs['pmin']
				idx_offset+=1
		else:
			ph_pars[i+idx_offset] = ( pta.params[i].prior.func_kwargs['pmax'] - pta.params[i].prior.func_kwargs['pmin'])*theta[i+idx_offset] + pta.params[i].prior.func_kwargs['pmin']
		if verbose:
			print(nm, pta.params[i].prior.func_args, pta.params[i].prior.func_kwargs['pmin'], pta.params[i].prior.func_kwargs['pmax'])
	return ph_pars

def log_l(x):
	return pta.get_lnlikelihood(x)

ndim = len(pta.params)
# initialize sampler with pool with pre-defined queue

Npar = len(pta.params)

#sampler = dynesty.NestedSampler(log_l, TransformPrior, ndim=Npar, nlive=500, bound='multi', sample='rwalk')

#sampler.run_nested(dlogz=0.1)
xs = {par.name: par.sample() for par in pta.params}
#print(xs)
# # dimension of parameter space
#ndim = len(xs)

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2) #Создаёт диагональную матрицу размерности ndim x ndim с элементами 0.0001задаёт начальный размер "шага" сэмплера по каждому параметру 

# # set up jump groups by red noise groups
ndim = len(xs)
#groups  = [range(0, ndim)]#создаёт группу из всех параметров
#groups.extend([[0,1]])#добавляет отдельную группу для параметров с индексами 1 и 2, Ускоряет сходимость, обновляя коррелированные параметры (например, амплитуду и спектральный индекс) совместно.
groups = [list(np.arange(0, ndim))]
psr_groups = sampler.get_psr_groups(pta)
groups.extend(psr_groups)
# intialize sampler
sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups, outDir=outdir)
N = 100000000 #общее количество итераций MCMC
x0 = np.hstack(p.sample() for p in pta.params) 
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)
chain = np.genfromtxt(outdir+"chain_1.txt")
burn = int(0.25 * chain.shape[0])
chain = chain[burn:, :]
truths = [13/3, np.log10(3e-14)]#задаются эталонные значения параметров для сравнения с результатами MCMC.
corner.corner(chain[burn:,-6:-4], 30, truths=truths);
#plt.show()