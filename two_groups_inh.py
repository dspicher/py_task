import json

import numpy as np
from brian2 import *
from IPython import embed
from scipy.stats import bernoulli


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

    return indices, times

def get_chain_spikes(input_rate, cycle_dur=100*ms, N_input=100):
    indices = range(N_input)
    times = arange(0, cycle_dur / ms, cycle_dur / ms / N_input) * ms
    chain = SpikeGeneratorGroup(N_input, indices, times, period=cycle_dur)
    return SpikeGeneratorGroup(N_input, indices, times, period=cycle_dur)


def simulate(neuron):
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

    dt_teacher = 1 * second
    ns['g_E'] = TimedArray(np.zeros(1) * msiemens, dt_teacher)
    ns['g_I'] = TimedArray(np.zeros(1) * msiemens, dt_teacher)

    ns['C'] = 1e-6 * farad

    nrn_eqs = '''
    #Teaching
    I_som = -g_E(t)*(U - E_E) - g_I(t)*(U - E_I) : amp

    #Spike currents
    #I_spike = 0.0*uamp : amp
    I_spike = - g_Na*(U - E_Na)*int((t-lastspike) < t_rise) - g_K*(U - E_K)*int((t-lastspike) > t_rise)*int((t-lastspike) < t_fall) : amp

    #Somatic potential
    dU/dt = -g_L*(U-E_L)/C - g_D*(U - V)/C - g_E_S_tot*(U - E_E)/C - g_I_S_tot*(U - E_I)/C + I_som/C + I_spike/C : volt

    #Dendritic potential
    dV/dt = -g_L*(V-E_L)/C - (g_E_D_tot + g_E_D_tot2)*(V - E_E)/C - g_S*(V-U)/C : volt

    #Dendritic prediction
    dV_star/dt = -g_L*(V_star - E_L)/C - g_D*(V_star - V)/C : volt

    #sum of excitatory synaptic conductance onto dendrite
    g_E_D_tot : siemens

    #sum of excitatory synaptic conductance onto dendrite
    g_E_D_tot2 : siemens

    g_E_S_tot : siemens
    g_I_S_tot : siemens
    '''

    group_size = 20
    NoI = NeuronGroup(group_size, nrn_eqs, threshold='phi(U, alpha, beta, r_max)*dt>rand()',
                      refractory=5 * ms)
    NoI.U = -70.0 * mV
    NoI.V = -70.0 * mV

    eqs_syn_som_exc = '''
    #Input with weight
    dg_E_S/dt = -g_E_S/tau_s : siemens

    g_E_S_tot_post = g_E_S : siemens (summed)

    w : 1
    '''

    s_som = Synapses(NoI, NoI, eqs_syn_som_exc, pre="g_E_S += w*msiemens")
    s_som.connect('i!=j and ((i < group_size/2) == (j < group_size/2))')
    s_som.w = 1e-1

    eqs_syn_som_inh = '''
    #Input with weight
    dg_I_S/dt = -g_I_S/tau_s_inh : siemens

    g_I_S_tot_post = g_I_S : siemens (summed)

    w : 1
    '''

    s_som_i = Synapses(NoI, NoI, eqs_syn_som_inh, pre="g_I_S += w*msiemens")
    s_som_i.connect('(i < group_size/2) != (j < group_size/2)')
    s_som_i.w = 1e-1

    # equations for the synapses
    eqs_syn = '''
    #Input without weight
    dg_s/dt = -g_s/tau_s : siemens

    #Input with weight
    dg_E_D/dt = -g_E_D/tau_s : siemens

    g_E_D_tot_post = g_E_D : siemens (summed)

    #Derivatives w.r.t the synaptic weight
    ddV_dw/dt = -(g_L + g_S + g_E_D)*dV_dw/C + g_S*dV_star_dw/C + (E_E - V)*g_s/C : volt

    ddV_star_dw/dt = -(g_L + g_D)*dV_star_dw/C + g_D*dV_dw/C : volt

    ddelta/dt = ((int((t-lastspike) < 0.11*ms)/(dt/second) - phi(V_star, alpha, beta, r_max)) * phi_prime(V_star, alpha, beta, r_max) / phi(V_star, alpha, beta, r_max) * dV_star_dw - delta)/tau_delta : 1

    dw/dt = eta*delta : 1

    eta : 1
    '''


    pre = '''
    g_s += 1.*msiemens
    g_E_D += w*msiemens
    '''

    N = 200
    cycle = 200*ms
    indices1, times1 = get_poisson_spikes(10*Hz, cycle, N)
    indices2, times2 = get_poisson_spikes(10*Hz, cycle, N)
    indices, times = (indices1, indices2), (times1, times2)

    N_presentations = 200
    p_pattern0 = 0.5
    patterns = bernoulli.rvs(p_pattern0, size=N_presentations)

    all_indices = []
    all_times = []
    for pt_idx, pt in enumerate(patterns):
        all_indices = all_indices + indices[pt]
        all_times = all_times + [pt_idx*cycle/ms + tm/ms for tm in times[pt]]
    all_times = np.array(all_times)*ms

    poisson = SpikeGeneratorGroup(N, all_indices, all_times)

    syns = Synapses(poisson, NoI, eqs_syn, pre=pre)
    syns.connect(True)
    syns.eta = 1e-3 / N
    syns.w = 3e-1 / N


    min_weights = np.zeros(N*group_size)

    @network_operation(when='end', dt=20 * ms)
    def clip_weights():
        syns.w = np.maximum(syns.w, min_weights)

    M = StateMonitor(NoI, ['U', 'V', 'V_star'], record=True, dt=1*ms)
    SM = SpikeMonitor(NoI)
    SynMon1 = StateMonitor(syns, ['w'], record=syns[:,group_size/2:], dt=2*ms)
    SynMon2 = StateMonitor(syns, ['w'], record=syns[:,:group_size/2], dt=2*ms)
    ratem = PopulationRateMonitor(NoI)


    durat = N_presentations*cycle
    run(durat, namespace=ns, report='stdout')


    figure(figsize=(20, 6))
    subplot(1, 3, 1)
    plot(M.t / ms, M.U.T / mV)
    #plot(M.t / ms, M.V.T / mV)
    subplot(1, 3, 2)
    plot(SynMon1.t / ms, SynMon1.w.T)
    subplot(1, 3, 3)
    plot(SynMon2.t / ms, SynMon2.w.T)

    figure()
    hist(concatenate([SM.spike_trains()[i]/second for i in range(group_size/2)]),bins=arange(0,durat/second+1), alpha=0.5)
    hist(concatenate([SM.spike_trains()[i]/second for i in range(group_size/2,group_size)]),bins=arange(0,durat/second+1), alpha=0.5)
    show()
    """
    import cPickle
    cPickle.dump((M.t/ms, M.U/mV, M.V/mV, SynMon.t/ms, SynMon.w.T, SM.spike_trains()), open('phis_al_{0}_b_{1}_rm_{2}.p'.format(nrn["alpha"], nrn["beta"], nrn["r_max"]),'wb'))
    """
    embed()

nrn = json.load(open('default_neuron.json', 'r'))
nrn["E_I"] = -75.0
simulate(nrn)
