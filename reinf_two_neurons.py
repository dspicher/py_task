import json
import warnings
from collections import OrderedDict

import numpy as np
from brian2 import *
from IPython import embed
from scipy.stats import bernoulli

from helper import do, dump


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

def get_poisson_spikes(input_rate=10*Hz, cycle_dur=100*ms, N_input=100, use_indices=None):
    if use_indices == None:
        use_indices = range(N_input)
    spike_nrs = np.random.poisson(input_rate * cycle_dur, len(use_indices))
    indices = []
    times = np.array([])
    for i, nr in enumerate(spike_nrs):
        indices = indices + [use_indices[i] for _ in range(nr)]
        times = np.concatenate((times, cycle_dur / ms * np.random.rand(nr)))
    times = times * ms

    return indices, times

def get_chain_spikes(input_rate, cycle_dur=100*ms, N_input=100):
    indices = range(N_input)
    times = arange(0, cycle_dur / ms, cycle_dur / ms / N_input) * ms
    chain = SpikeGeneratorGroup(N_input, indices, times, period=cycle_dur)
    return SpikeGeneratorGroup(N_input, indices, times, period=cycle_dur)


def simulate((repetition_i, p)):
    neuron = json.load(open('default_neuron.json', 'r'))
    neuron["g_adap_delta"] = p["g_adap_delta"]
    neuron["tau_adap"] = p["tau_adap"]
    neuron["r_max"] = p["r_max"]
    neuron["tau_s_inh"] = p["tau_exc_inh"]
    neuron["tau_s_exc"] = p["tau_exc_inh"]
    if "tau_delta" in p.keys():
        neuron["tau_delta"] = p["tau_delta"]
    np.random.seed(p.get("seed",42))  # seed for pseudo-randomness

    ns = {}
    for key, value in neuron.items():
        if key[0] == 'g':
            unit = msiemens
        elif key[0] == 'E':
            unit = mV
        elif key[0] == 't':
            unit = ms
        elif key[0] == 'I':
            unit = uamp
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
    ns["rho_E"] = p["rho_E"]
    ns["rho_I"] = p["rho_I"]
    defaultclock.dt = 0.05*ms

    dt_teacher = 1 * second
    ns['g_E'] = TimedArray(np.zeros(1) * msiemens, dt_teacher)
    ns['g_I'] = TimedArray(np.zeros(1) * msiemens, dt_teacher)

    ns['C'] = 1e-6 * farad

    nrn_eqs = '''
    #Teaching
    I_som = -g_E(t)*(U - E_E) - g_I(t)*(U - E_I) : amp

    #Spike currents
    I_spike = 0.0*uamp : amp
    #I_spike = - g_Na*(U - E_Na)*int((t-lastspike) < t_rise) - g_K*(U - E_K)*int((t-lastspike) > t_rise)*int((t-lastspike) < t_fall) : amp

    #Adaptation currents
    dg_adap/dt = -g_adap/tau_adap : siemens
    I_adap = - g_adap*(U - E_adap) : amp

    #Intra-group connections
    I_intra = - g_E_S_tot*(U - E_E) - g_E_S_tot*rho_I*(U - E_I) : amp

    #Inter-group connections
    I_inter = - g_I_S_tot*(U - E_I) - g_I_S_tot*rho_E*(U - E_E) : amp

    #Dendritic currents
    I_den = - g_D*(U - V) : amp

    #Leak current
    I_L = -g_L*(U - E_L) : amp

    #Somatic potential
    dU/dt = I_L/C + I_den/C  + I_intra/C + I_inter/C  + I_adap/C + I_som/C + I_spike/C : volt

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

    group_size = 1
    NoI = NeuronGroup(2*group_size, nrn_eqs, threshold='phi(U, alpha, beta, r_max)*dt>rand()',
                      refractory=0.01*ms, reset='g_adap += g_adap_delta/(tau_adap/ms)')
    NoI.U = -70.0 * mV
    NoI.V = -70.0 * mV

    eqs_syn_som_exc = '''
    #Input with weight
    dg_E_S/dt = -g_E_S/tau_s_exc : siemens

    g_E_S_tot_post = g_E_S : siemens (summed)

    w : 1
    '''

    s_som_e = Synapses(NoI, NoI, eqs_syn_som_exc, pre="g_E_S += w/(tau_s_exc/ms)*msiemens")
    #s_som_e.connect('i!=j and ((i < group_size) == (j < group_size))')
    s_som_e.connect('(i < group_size) == (j < group_size)')
    s_som_e.w = p["we"]

    eqs_syn_som_inh = '''
    #Input with weight
    dg_I_S/dt = -g_I_S/tau_s_inh : siemens

    g_I_S_tot_post = g_I_S : siemens (summed)

    w : 1
    '''

    s_som_i = Synapses(NoI, NoI, eqs_syn_som_inh, pre="g_I_S += w/(tau_s_inh/ms)*msiemens")
    s_som_i.connect('(i < group_size) != (j < group_size)')
    s_som_i.w = p["wi"]

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

    #ddelta/dt = ((int((t-lastspike) < 0.11*ms)/(dt/second) - phi(V_star, alpha, beta, r_max)) * phi_prime(V_star, alpha, beta, r_max) / phi(V_star, alpha, beta, r_max) * dV_star_dw - delta)/tau_delta : 1
    ddelta/dt = ((phi(U_post, alpha, beta, r_max) - phi(V_star, alpha, beta, r_max)) * phi_prime(V_star, alpha, beta, r_max) / phi(V_star, alpha, beta, r_max) * dV_star_dw - delta)/tau_delta : 1
    w : 1

    eta : 1
    '''


    on_pre = '''
    g_s += 1.*msiemens
    g_E_D += w*msiemens
    '''
    f_in = 10*Hz
    N_input = 100
    cycle = 500*ms
    used_indices1 = range(int((1 + p["overlap"])*N_input/2))
    used_indices2 = range(int((1 - p["overlap"])*N_input/2), N_input)
    indices1, times1 = get_poisson_spikes(f_in, cycle, N_input, use_indices = used_indices1)
    indices2, times2 = get_poisson_spikes(f_in, cycle, N_input, use_indices = used_indices2)
    indices, times = (indices1, indices2), (times1, times2)

    N_presentations = 150
    p_pattern0 = 0.5
    patterns = bernoulli.rvs(1.0-p_pattern0, size=N_presentations)


    durat = len(patterns)*cycle

    all_indices = []
    all_times = []
    for pt_idx, pt in enumerate(patterns):
        all_indices = all_indices + indices[pt]
        all_times = all_times + [pt_idx*cycle/ms + tm/ms for tm in times[pt]]
    all_times = np.array(all_times)*ms

    poisson = SpikeGeneratorGroup(N_input, all_indices, all_times)

    syns = Synapses(poisson, NoI, eqs_syn, pre=on_pre)
    syns.connect(True)
    syns.eta = p["eta"] / N_input
    w_base = 1e0 / N_input

    sigma = w_base*p["sigmaf"]

    syns.w = w_base + sigma*np.random.randn(2*N_input*group_size)

    min_weights = np.zeros(N_input*group_size*2)

    @network_operation(when='end', dt=cycle)
    def update_weights(t):
        pattern_count = int(t/us) / int(cycle/us) - 1
        ts1 = np.array([SM.t[idx]/ms for idx in range(np.array(SM.t).shape[0]) if SM.i[idx] in range(group_size)])
        ts2 = np.array([SM.t[idx]/ms for idx in range(np.array(SM.t).shape[0]) if SM.i[idx] in range(group_size,2*group_size)])
        spikes1 = ts1[ts1 >= (t-cycle)/ms]
        spikes2 = ts2[ts2 >= (t-cycle)/ms]

        #group zero wins for pattern 0, group 1 for pattern 1
        #check mapping!!
        if spikes1.shape[0] == spikes2.shape[0]:
            reward = 0.0
        elif spikes1.shape[0] > spikes2.shape[0]:
            reward = 2*(1-patterns[pattern_count])-1
        else:
            reward = 2*patterns[pattern_count]-1

        # flip after half
        if t >= durat/3:
            reward = -reward

        # clip weights at zero
        syns.w = np.maximum(syns.w + defaultclock.dt/ms*syns.eta*syns.delta*reward, np.zeros(np.array(syns.w).shape))

    M = StateMonitor(NoI, ['U', 'V', 'V_star', 'I_adap','I_den', 'I_L',  'g_adap', 'I_inter', 'I_intra'], record=True, dt=2*ms)
    SM = SpikeMonitor(NoI)
    SynMon = StateMonitor(syns, ['w'], record=True, dt=10*ms)
    ratem = PopulationRateMonitor(NoI)



    run(durat, namespace=ns, report='text')

    hertz_factor = 1000/(cycle/ms)*1.0/(group_size)

    t_s = np.array(SM.t)
    i_s = np.array(SM.i)
    spikes1 = concatenate([t_s[i_s==i]/ms for i in range(group_size)])
    spikes2 = concatenate([t_s[i_s==i]/ms for i in range(group_size,group_size*2)])
    all_spikes = [spikes1, spikes2]
    pattern_starts = arange(0,len(patterns)*cycle/ms,cycle/ms)
    assigned_spikes = {}

    spike_counts = np.zeros((pattern_starts.shape[0],2))
    for i, ps in enumerate(pattern_starts):
        spike_counts[i, 0] = hertz_factor*np.sum(np.logical_and(spikes1 >= ps, spikes1 < ps+cycle/ms))
        spike_counts[i, 1] = hertz_factor*np.sum(np.logical_and(spikes2 >= ps, spikes2 < ps+cycle/ms))

    for pattern in [0,1]:
        for group_idx, spikes in enumerate(all_spikes):
            starts = reshape(pattern_starts[patterns==pattern],(-1,1))
            diffs = spikes - starts
            found_idxs = np.sum(logical_and(diffs > 0, diffs <= cycle/ms),0)
            assert np.all(found_idxs <= 1)
            assigned_spikes[(group_idx, pattern)] = spikes[found_idxs==1]
    n_spikes = {k:1.0*assigned_spikes[k].shape[0] for k in assigned_spikes.keys()}

    """
    fig = figure()
    plot(M.t / ms, M.U.T[:,:group_size] / mV, 'blue', alpha=0.5)
    plot(M.t / ms, M.U.T[:,group_size:] / mV, 'green', alpha=0.5)
    plot(M.t / ms, M.V.T[:,:group_size] / mV, 'b--', alpha=0.5)
    plot(M.t / ms, M.V.T[:,group_size:] / mV, 'g--', alpha=0.5)
    plt.show()

    """

    fig = figure(figsize=(6, 10))
    w_end = reshape(SynMon.w[:,-1],(group_size*2,N_input), order='F')
    imshow(w_end.T,aspect='auto',interpolation='nearest',cmap='Oranges')
    colorbar()
    savefig(p["ident"]+"_weights.pdf")
    plt.close(fig)

    ws=reshape(SynMon.w,(group_size*2,N_input,-1),order='F')


    colors=['blue', 'green']
    fig = figure(figsize=(12,5))
    for pattern in [0,1]:
        for group_idx in [0,1]:
            hist, bins = np.histogram(assigned_spikes[(group_idx,pattern)],bins=arange(0,durat/ms+1,cycle/ms))
            # pattern 0: below zero, pattern 1: above zero
            # => for reinforcement learning task: blue should be below, green above
            factor = hertz_factor*(pattern*2-1)
            try:
                width = 0.9*(bins[1]-bins[0])
            except:
                width = 1.0
            center = (bins[:-1] + bins[1:]) / 2
            plt.bar(center, factor*hist, align='center', width=width, color=colors[group_idx], alpha=0.5)

    savefig(p["ident"]+"_spikes.pdf")
    plt.close(fig)

    fig = figure(figsize=(6,6))
    colors = ['b', 'g']
    for idx, pat in enumerate(patterns):
        scatter(spike_counts[idx,0], spike_counts[idx,1], c=colors[pat], lw=0, s=20, alpha=1.0/N_presentations*(idx+1))
    xl, yl = xlim(), ylim()
    plot([0,1000], [0,1000], 'k--')
    xlim(xl)
    ylim(yl)
    savefig(p["ident"]+"_scatter.pdf")
    plt.close(fig)

    fig = figure()
    for i in range(N_input/2):
        plot(SynMon.t/ms,ws[0,i,:], 'b', alpha=0.2)
        plot(SynMon.t/ms,ws[1,i,:], 'g', alpha=0.2)
    plot(SynMon.t/ms, np.mean(np.squeeze(ws[0,:N_input/2,:]),0), 'b', lw=3)
    plot(SynMon.t/ms, np.mean(np.squeeze(ws[1,:N_input/2,:]),0), 'g', lw=3)
    savefig(p["ident"]+"_weights_time1.pdf")
    plt.close(fig)

    fig = figure()
    for i in range(N_input/2,N_input):
        plot(SynMon.t/ms,ws[0,i,:], 'b', alpha=0.2)
        plot(SynMon.t/ms,ws[1,i,:], 'g', alpha=0.2)
    plot(SynMon.t/ms, np.mean(np.squeeze(ws[0,N_input/2:,:]),0), 'b', lw=3)
    plot(SynMon.t/ms, np.mean(np.squeeze(ws[1,N_input/2:,:]),0), 'g', lw=3)
    savefig(p["ident"]+"_weights_time2.pdf")
    plt.close(fig)
    """
    fig = figure()
    plot(M.t, M.U.T)
    yl = ylim()
    for idx, spiker in enumerate(array(SM.i)):
        if spiker == 0:
            plot([SM.t[idx], SM.t[idx]], yl, 'b', alpha=0.1)
        else:
            plot([SM.t[idx], SM.t[idx]], yl, 'g', alpha=0.1)
    ylim(yl)
    savefig(p["ident"]+"_voltage.pdf")
    plt.close(fig)
    """

    fig = figure(figsize=(20,4))
    ax1 = plt.subplot(151)
    plot(M.t/ms, M.I_den.T)
    title("dend")

    ax2 = plt.subplot(152, sharey=ax1)
    plot(M.t/ms, M.I_inter.T)
    title("inter")

    ax3 = plt.subplot(153, sharey=ax1)
    plot(M.t/ms, M.I_intra.T)
    title("intra")

    ax4 = plt.subplot(154, sharey=ax1)
    plot(M.t/ms, M.I_adap.T)
    title("Iadap")

    ax4 = plt.subplot(155)
    plot(M.t/ms, M.g_adap.T)
    title("gadap")
    savefig(p["ident"]+"_currents.pdf")
    plt.close(fig)
    #plt.show()

    #results = (SynMon1.t/ms, SynMon1.w, SynMon2.w, SM.spike_trains(), assigned_spikes, selectivities)
    #dump(results, p["ident"])

params = OrderedDict()
params["g_adap_delta"] = [1.3, 1.5]
params["tau_adap"] = [1000.0]
params["we"] = [6e-1]#, 1e-1]
params["wi"] = [1e0]
params["rho_E"] = [0.0]
params["rho_I"] = [2.0]
params["r_max"] = [1.0]
params["tau_exc_inh"] = [20.0]#, 200.0]
params["sigmaf"] = [0.1]#, 0.3]
params["eta"] = [2e-1]#,5e-4, 1e-3, 2e-3, 5e-3]#, 2e-3, 4e-3]
params["overlap"] = [0.0, 0.25]
params["seed"] = range(2)
params["tau_delta"] = [500.0]
file_prefix = 'flip'

do(simulate, params, file_prefix, withmp=True, create_notebooks=False)
