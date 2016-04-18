import json

import numpy as np
from brian2 import *
from IPython import embed


@implementation('cpp', '''
     #include<math.h>
     double phi(double U, double alpha, double beta, double r_max) {
        return r_max / (1 + exp(-beta*(U-alpha)));
     }
     ''')
@check_units(U=volt, alpha=volt, beta=volt**-1, r_max=second**-1, result=second**-1)
def phi(U, alpha, beta, r_max):
    pass


@implementation('cpp', '''
     #include<math.h>
     double phi_prime(double U, double alpha, double beta, double r_max) {
        double num = exp((U + alpha) * beta) * r_max * beta;
        double denom = pow(exp(U * beta) + exp(alpha * beta), 2.0);
        return num / denom;
     }
     ''')
@check_units(U=volt, alpha=volt, beta=volt**-1, r_max=second**-1, result=(second * volt)**-1)
def phi_prime(U, alpha, beta, r_max):
    pass

def get_poisson_spikes(input_rate=10*Hz, cycle_dur=100*ms, N_input=100):
    spike_nrs = np.random.poisson(input_rate * cycle_dur, N_input)
    indices = []
    times = np.array([])
    for i, nr in enumerate(spike_nrs):
        indices = indices + [i for _ in range(nr)]
        times = np.concatenate((times, cycle_dur / ms * np.random.rand(nr)))
    times = times * ms

    poisson = SpikeGeneratorGroup(N_input, indices, times, period=cycle_dur)

    return poisson

def get_chain_spikes(input_rate=10*Hz, cycle_dur=100*ms, N_input=100):
    indices = range(N_input)
    times = arange(0, cycle_dur / ms, cycle_dur / ms / N_input) * ms
    chain = SpikeGeneratorGroup(N_input, indices, times, period=cycle_dur)
    return SpikeGeneratorGroup(N_input, indices, times, period=cycle_dur)


def simulate(neuron, U):
    np.random.seed(42)  # seed for pseudo-randomness

    ns = {}
    for key, value in neuron.items():
        if key[0] == 'g':
            unit = msiemens
        elif key[0] == 'E':
            unit = mV
        elif key[0] == 't':
            unit = ms
        elif key == 'alpha':
            unit = mV
        elif key == 'beta':
            unit = mV**-1
        elif key == 'r_max':
            unit = ms**-1
        else:
            raise Exception()
        ns[key] = value * unit


    ns['phi_prime'] = phi_prime
    ns['phi'] = phi
    defaultclock.dt = 0.05*ms

    nrn_eqs = '''

    #Somatic potential
    dU/dt = 0*mV/second : volt
    '''

    group_size = 1
    NoI = NeuronGroup(group_size, nrn_eqs, threshold='phi(U, alpha, beta, r_max)*dt>rand()',
                      refractory=0 * ms)
    NoI.U = U * mV

    #M = StateMonitor(NoI, ['U', 'V', 'V_star'], record=True, dt=1*ms)
    SM = SpikeMonitor(NoI, variables='U')
    #SynMon = StateMonitor(s_chain, ['w', 'dV_dw', 'dV_star_dw', 'g_s', 'g_E_D', 'delta'], record=True, dt=2*ms)

    run(60*second, namespace=ns, report='stdout')
    import cPickle
    cPickle.dump((SM.values('U'), SM.spike_trains()), open('phis_al_{0}_b_{1}_rm_{2}_U_{3}.p'.format(nrn["alpha"], nrn["beta"], nrn["r_max"], U),'wb'))

nrn = json.load(open('default_neuron.json', 'r'))
for alpha in [-50.0, -55.0]:
    for beta in [0.2, 0.5]:
        for r_max in [0.1, 0.3]:
            nrn["alpha"] = alpha
            nrn["beta"] = beta
            nrn["r_max"] = r_max
            for U in arange(-70.0, -31.0, 5.0):
                simulate(nrn, U)
