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

def get_poisson_spikes(input_rate=6*Hz, cycle_dur=100*ms, N_input=100):
    spike_nrs = np.random.poisson(input_rate * cycle_dur, N_input)
    indices = []
    times = np.array([])
    for i, nr in enumerate(spike_nrs):
        indices = indices + [i for _ in range(nr)]
        times = np.concatenate((times, cycle_dur / ms * np.random.rand(nr)))
    times = times * ms

    return indices, times

def get_chain_spikes(input_rate=10*Hz, cycle_dur=100*ms, N_input=100):
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
    dU/dt = -g_L*(U-E_L)/C - g_D*(U - V)/C - g_E_S_tot*(U - E_E)/C + I_som/C + I_spike/C : volt

    #Dendritic potential
    dV/dt = -g_L*(V-E_L)/C - (g_E_D_tot + g_E_D_tot2)*(V - E_E)/C - g_S*(V-U)/C : volt

    #Dendritic prediction
    dV_star/dt = -g_L*(V_star - E_L)/C - g_D*(V_star - V)/C : volt

    #sum of excitatory synaptic conductance onto dendrite
    g_E_D_tot : siemens

    #sum of excitatory synaptic conductance onto dendrite
    g_E_D_tot2 : siemens

    #sum of excitatory synaptic conductance onto dendrite
    g_E_S_tot : siemens
    '''

    group_size = 5
    NoI = NeuronGroup(group_size, nrn_eqs, threshold='phi(U, alpha, beta, r_max)*dt>rand()',
                      refractory=10 * ms)
    NoI.U = -70.0 * mV
    NoI.V = -70.0 * mV

    eqs_syn_som = '''
    #Input with weight
    dg_E_S/dt = -g_E_S/tau_s : siemens

    g_E_S_tot_post = g_E_S : siemens (summed)

    w : 1
    '''

    s_som = Synapses(NoI, NoI, eqs_syn_som, pre="g_E_S += w*msiemens")
    s_som.connect('i!=j')
    s_som.w = 1e-1

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

    # equations for the synapses
    eqs_syn2 = '''
    #Input without weight
    dg_s/dt = -g_s/tau_s : siemens

    #Input with weight
    dg_E_D/dt = -g_E_D/tau_s : siemens

    g_E_D_tot2_post = g_E_D : siemens (summed)

    w : 1
    '''


    pre = '''
    g_s += 1.*msiemens
    g_E_D += w*msiemens
    '''

    N = 100
    cycle = 200*ms
    indices, times = get_poisson_spikes(cycle_dur=cycle, N_input=N)
    poisson = SpikeGeneratorGroup(N, indices, times, period=cycle)

    indices2, times2 = get_poisson_spikes(cycle_dur=cycle, N_input=N)
    poisson2 = SpikeGeneratorGroup(N, indices2, times2, period=cycle)


    syns = Synapses(poisson, NoI, eqs_syn, pre=pre)
    syns.connect(True)
    syns.w = 0.0
    syns.eta = 0.0


    syns2 = Synapses(poisson2, NoI, eqs_syn2, pre=pre)
    syns2.connect(True)
    syns2.w = 3e-1 / N


    min_weights = np.zeros(N*group_size)

    @network_operation(when='end', dt=20 * ms)
    def clip_weights():
        syns.w = np.maximum(syns.w, min_weights)

    M = StateMonitor(NoI, ['U', 'V', 'V_star'], record=True, dt=1*ms)
    SM = SpikeMonitor(NoI, variables='U')
    SynMon = StateMonitor(syns, ['w'], record=True, dt=2*ms)
    SynMon2 = StateMonitor(syns2, ['w'], record=True, dt=2*ms)
    ratem = PopulationRateMonitor(NoI)

    borders = []

    borders.append(defaultclock.t)
    durat = 10*cycle
    run(durat, namespace=ns, report='stdout')

    syns.w = 3e-1 / N
    syns2.w = 0.0
    borders.append(defaultclock.t)
    durat = 10*cycle
    run(durat, namespace=ns, report='stdout')


    for i in range(10):
        syns.eta = 1e-3 / N
        borders.append(defaultclock.t)
        durat = 10*second
        run(durat, namespace=ns, report='stdout')

        syns.eta = 0.0
        borders.append(defaultclock.t)
        durat = 10*cycle
        run(durat, namespace=ns, report='stdout')


        syns2.w = syns.w
        syns.w = 0.0
        borders.append(defaultclock.t)
        durat = 10*cycle
        run(durat, namespace=ns, report='stdout')

        syns.w = syns2.w
        syns2.w = 0



    borders.append(defaultclock.t)
    borders = np.array(borders/ms)


    figure(figsize=(20, 6))
    subplot(1, 3, 1)
    plot(M.t / ms, M.U.T / mV)
    #plot(M.t / ms, M.V.T / mV)
    subplot(1, 3, 2)
    plot(SynMon.t / ms, SynMon.w.T)
    subplot(1, 3, 3)
    plot(SynMon2.t / ms, SynMon2.w.T)

    figure()
    window = 500*ms
    window_length = int(window/defaultclock.dt)
    cumsum = numpy.cumsum(numpy.insert(ratem.rate, 0, 0))
    binned_rate = (cumsum[window_length:] - cumsum[:-window_length]) / window_length
    ts = np.array(ratem.t[window_length-1:]/ms)

    lims = (borders[0], borders[1])
    mask = logical_and(ts >= lims[0], ts <= lims[1])
    plot(ts[mask], binned_rate[mask], 'g')

    lims = (borders[1], borders[2])
    mask = logical_and(ts >= lims[0], ts <= lims[1])
    plot(ts[mask], binned_rate[mask], 'b--')

    i = 2
    while i < len(borders) - 1:
        lims = (borders[i], borders[i+1])
        mask = logical_and(ts >= lims[0], ts <= lims[1])
        plot(ts[mask], binned_rate[mask], 'b')
        i += 1

        lims = (borders[i], borders[i+1])
        mask = logical_and(ts >= lims[0], ts <= lims[1])
        plot(ts[mask], binned_rate[mask], 'b--')
        i += 1

        lims = (borders[i], borders[i+1])
        mask = logical_and(ts >= lims[0], ts <= lims[1])
        plot(ts[mask], binned_rate[mask], 'g')
        i += 1

    show()
    """
    import cPickle
    cPickle.dump((M.t/ms, M.U/mV, M.V/mV, SynMon.t/ms, SynMon.w.T, SM.spike_trains()), open('phis_al_{0}_b_{1}_rm_{2}.p'.format(nrn["alpha"], nrn["beta"], nrn["r_max"]),'wb'))
    """
    embed()

nrn = json.load(open('default_neuron.json', 'r'))
for alpha in [-50.0, -52.0, -54.0]:
    for beta in [0.2, 0.3, 0.5]:
        for r_max in [0.1, 0.3]:
            nrn["alpha"] = alpha
            nrn["beta"] = beta
            nrn["r_max"] = r_max
            simulate(nrn)
