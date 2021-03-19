'''
Explorations of the number of people with exposure to COVID-19 in various settings
'''
import sciris as sc
import covasim as cv
import numpy as np
import pandas as pd

########################################################################
# Settings and initialisation
########################################################################
# Check version
cv.check_save_version('3.0', die=True)
cv.git_info('covasim_version.json')

# Saving and plotting settings
debug = 0
do_plot = 1
do_save = 1
save_sim = 1
do_show = 0

to_plot = sc.objdict({
    'Cumulative diagnoses': ['cum_diagnoses'],
    'Cumulative deaths': ['cum_deaths'],
    'New diagnoses': ['new_diagnoses'],
    'New infections': ['new_infections'],
    'Cumulative infections': ['cum_infections'],
    'Cumulative symptomatic': ['cum_symptomatic'],
})

# Other settings
verbose = -1
seed = 1

# Define what/where to run
location = ['lucknow', 'wcape'][0]
runoptions = ['quickfit', # Does a quick preliminary calibration. Quick to run, ~30s
              'sweeps', # Parameter sweeps to calculate mismatch maps
              ]
whattorun = runoptions[1] # Select which of the above to run

# Filepaths
data_path = f'data/{location}.csv'
resfolder = 'results'
cachefolder = 'cache'

# Important dates
start_day = {'lucknow': '2020-03-21', 'wcape': '2020-03-30'}[location]
end_day = {'lucknow': '2021-02-28', 'wcape': '2020-10-31'}[location]


########################################################################
# Create the baseline simulation
########################################################################
def make_sim(location, p=None, pop_infected=None, beta=None, rel_symp_prob=None, symp_prob=None,
             start_day=None, end_day=None, data_path=None,
             seed=None, verbose=0, debug=1, meta=None):
    ''' Create sim'''
    print(f'Making sim {meta.inds} ({meta.count} of {meta.n_sims})...')
    if location=='lucknow':
        sim = make_lucknow(p=p, pop_infected=pop_infected, beta=beta, rel_symp_prob=rel_symp_prob, symp_prob=symp_prob,
                           start_day=start_day, end_day=end_day, data_path=data_path,
                           seed=seed, verbose=verbose, debug=debug, meta=meta)
    elif location=='wcape':
        sim = make_wcape(p=p, pop_infected=pop_infected, beta=beta, rel_symp_prob=rel_symp_prob, symp_prob=symp_prob,
                         start_day=start_day, end_day=end_day, data_path=data_path,
                         seed=seed, verbose=verbose, debug=debug, meta=meta)
    return sim


def make_lucknow(p=None, pop_infected=None, beta=None, rel_symp_prob=None, symp_prob=None,
                 start_day=None, end_day=None, data_path=None, seed=None, verbose=0, debug=1, meta=None):
    # Create parameters
    total_pop    = 3.5e6
    if p is not None:
        pop_infected=p.pop_infected
        beta=p.beta
        rel_symp_prob=p.rel_symp_prob
        symp_prob=p.symp_prob

    pars = sc.objdict(
        pop_size     = [100e3, 5e3][debug],
        pop_infected = pop_infected,
        rescale      = True,
        pop_type     = 'hybrid',
        start_day    = start_day,
        end_day      = end_day,
        rand_seed    = seed,
        verbose      = verbose,
        beta         = beta,
        rel_symp_prob = rel_symp_prob,
        iso_factor = dict(h=0.5, s=0.4, w=0.4, c=0.5), # default: dict(h=0.3, s=0.1, w=0.1, c=0.1)
        quar_factor= dict(h=0.9, s=0.5, w=0.5, c=0.7), # default: dict(h=0.6, s=0.2, w=0.2, c=0.2)
    )
    pars.pop_scale    = int(total_pop/pars.pop_size)

    # Create interventions
    interventions = []

    # Beta interventions
    interventions += [
        cv.clip_edges(days=['2020-03-22', '2020-09-01', '2021-02-01'],
                      changes=[0.1, 3/12, 1], layers=['s']),
        cv.change_beta(['2020-03-22'], [0.8], layers=['s']),
        cv.clip_edges(days=['2020-03-22', '2020-04-16', '2020-08-01', '2021-02-01'],
                      changes=[0.1, 0.5, 0.9, 1.0], layers=['w']),
        cv.change_beta(['2020-03-22'], [0.8], layers=['w']),
        cv.change_beta(['2020-03-22','2020-08-01','2021-02-01'], [0.6, 0.7, 0.8], layers=['c']),
        ]

    # Testing interventions
    interventions += [
        cv.test_prob(symp_prob=symp_prob, start_day=0, test_delay=5),
    ]

    # Create sim
    sim = cv.Sim(pars=pars, interventions=interventions, datafile=data_path, location='india')
    for intervention in sim['interventions']:
        intervention.do_plot = False

    sim.label = f'Lucknow {seed}'

    # Store metadata
    sim.meta = meta

    return sim


def make_wcape(p=None, start_day=None, end_day=None, data_path=None,
                 seed=None, verbose=0, debug=1, meta=None):

#    print(f'Making sim {meta.inds} ({meta.count} of {meta.n_sims})...')

    # Create parameters
    total_pop    = 7e6
    pars = sc.objdict(
        pop_size     = [100e3, 5e3][debug] ,
        pop_infected = 100,
        rescale      = True,
        pop_type     = 'hybrid',
        start_day    = start_day,
        end_day      = end_day,
        rand_seed    = seed,
        verbose      = verbose,
        beta         = p.beta,
        rel_symp_prob = p.rel_symp_prob,
        iso_factor = dict(h=0.5, s=0.4, w=0.4, c=0.5), # default: dict(h=0.3, s=0.1, w=0.1, c=0.1)
        quar_factor= dict(h=0.9, s=0.5, w=0.5, c=0.7), # default: dict(h=0.6, s=0.2, w=0.2, c=0.2)
    )
    pars.pop_scale = int(total_pop / pars.pop_size)

    # Create interventions
    interventions = []

    # Beta interventions
    interventions += [
        cv.clip_edges(days=['2020-03-18', '2020-06-09', '2020-07-27', '2020-08-24'],
                      changes=[0.1, 0.8, 0.1, 0.8], layers=['s']),
        cv.change_beta(['2020-06-09'], [0.35], layers=['s']),
        cv.clip_edges(days=['2020-03-27', '2020-05-01', '2020-06-01', '2020-08-17', '2020-09-20'],
                      changes=[0.65, 0.70, 0.72, 0.74, 0.92], layers=['w']),
        cv.change_beta(['2020-04-10'], [0.75], layers=['w']),
        cv.change_beta(['2020-04-10'], [0.75], layers=['c'], do_plot=False),  # Mandatory masks, then enforced
    ]

    # Testing interventions
    interventions += [
        cv.test_prob(symp_prob=p.symp_prob, start_day=0, test_delay=7),
        cv.contact_tracing(start_day='2020-03-01',
                           trace_probs={'h': 1, 's': 0.5, 'w': 0.5, 'c': 0.1},
                           trace_time={'h': 1, 's': 3, 'w': 7, 'c': 14})
    ]

    # Create sim
    sim = cv.Sim(pars=pars, interventions=interventions, datafile=data_path, location='south africa')
    for intervention in sim['interventions']:
        intervention.do_plot = False

    # Store metadata
    sim.meta = meta

    return sim


def run_sim(sim, do_save=True, do_shrink=True):
    ''' Run a simulation '''
    print(f'Running sim {sim.meta.count:5g} of {sim.meta.n_sims:5g} {str(sim.meta.vals.values()):40s}')
    sim.run()
    if do_shrink:
        sim.shrink()
    return sim


def make_msims(sims):
    ''' Take a slice of sims and turn it into a multisim '''
    msim = cv.MultiSim(sims)
    draw, seed = sims[0].meta.inds
    for s,sim in enumerate(sims): # Check that everything except seed matches
        assert draw == sim.meta.inds[0]
        assert (s==0) or seed != sim.meta.inds[1]
    msim.meta = sc.objdict()
    msim.meta.inds = [draw]
    msim.meta.vals = sc.dcp(sims[0].meta.vals)
    msim.meta.vals.pop('seed')
    print(f'Processing multisim {msim.meta.vals.values()}...')

    if save_sim: # NB, generates ~2 GB of files for a full run
        id_str = '_'.join([str(i) for i in msim.meta.inds])
        msimfile = f'{cachefolder}/final_msim{id_str}.msim' # File to save the partially run sim to
        msim.save(msimfile)

    return msim



########################################################################
# Run calibration and scenarios
########################################################################
if __name__ == '__main__':

    # # High transmissibility, low diagnosis rate -> lots of infections
    p_high = sc.objdict(
        pop_infected = 150,
        beta = 0.019,
        rel_symp_prob = 0.25,
        symp_prob = .025)

    # Low transmissibility, high diagnosis rate -> fewer infections
    p_low = sc.objdict(
        pop_infected = 150,
        beta = 0.015,
        rel_symp_prob = 1,
        symp_prob = .075)

    # Random draws
    def get_p(location):
        ''' Randomly drawn some parameters '''
        if location == 'lucknow':
            p = sc.objdict(
                pop_infected=150,
                beta=np.random.uniform(0.014, 0.02),
                rel_symp_prob=np.random.uniform(0.25, 1.0),
                symp_prob=np.random.uniform(0.025, 0.075),
            )
        return p

    which = 'draw'

    # Quick calibration
    if whattorun=='quickfit':
        sims = []
        n_runs = 4
        if which == 'high':     p = p_high
        elif which == 'low':    p = p_low
        elif which == 'draw':   p = get_p(location)

        for i in range(n_runs):
            s = make_sim(location, p=p, start_day=start_day, end_day=end_day, data_path=data_path,
                         seed=seed+i, verbose=verbose, debug=debug, meta=None)
            sims.append(s)
        msim = cv.MultiSim(sims)
        msim.run()
        msim.reduce()
        if do_plot:
              msim.plot(to_plot=to_plot, do_save=True, do_show=False, fig_path=f'{location}_{which}.png',
                     legend_args={'loc': 'upper left'}, axis_args={'hspace': 0.4}, interval=70, n_cols=2)


    # Quick calibration
    elif whattorun=='sweeps':

        n_seeds = [10, 2][debug]
        n_draws = [2000, 10][debug]
        n_sims = n_seeds * n_draws
        sims_file = f'{cachefolder}/all_sims.obj'
        count = 0
        ikw = []
        T = sc.tic()

        # Make sims
        sc.heading('Making sims...')
        for draw in range(n_draws):
            p = get_p(location)
            for seed in range(n_seeds):
                print(f'Creating arguments for sim {count} of {n_sims}...')
                count += 1
                meta = sc.objdict()
                meta.count = count
                meta.n_sims = n_sims
                meta.inds = [draw, seed]
                meta.vals = sc.objdict(sc.mergedicts(p, dict(seed=seed)))
                ikw.append(sc.dcp(meta.vals))
                ikw[-1].meta = meta

        kwargs = dict(location=location, start_day=start_day, end_day=end_day, data_path=data_path,
                      verbose=verbose, debug=debug)
        sim_configs = sc.parallelize(make_sim, iterkwargs=ikw, kwargs=kwargs)
        if do_save:
            cv.save(filename=sims_file, obj=sim_configs)

        # Run sims
        all_sims = sc.parallelize(run_sim, iterarg=sim_configs, kwargs=dict(do_save=do_save))
        sims = np.empty((n_draws, n_seeds), dtype=object)
        for sim in all_sims: # Unflatten array
            draw, seed = sim.meta.inds
            sims[draw, seed] = sim

        # Convert to msims
        all_sims_semi_flat = []
        for draw in range(n_draws):
            sim_seeds = sims[draw, :].tolist()
            all_sims_semi_flat.append(sim_seeds)
        msims = np.empty(n_draws, dtype=object)
        all_msims = sc.parallelize(make_msims, iterarg=all_sims_semi_flat)
        for msim in all_msims: # Unflatten array
            draw = msim.meta.inds
            msims[draw] = msim

        # Calculate mismatches
        d = sc.objdict()
        d.beta = []
        d.rel_symp_prob = []
        d.symp_prob = []
        d.mismatch = []
        d.infections = []
        d.test_prob = []
        for msim in all_msims:
            d.test_prob.append(msim.meta.vals.rel_symp_prob * (1-(1-msim.meta.vals.symp_prob)**10))
            d.beta.append(msim.meta.vals.beta)
            d.rel_symp_prob.append(msim.meta.vals.rel_symp_prob)
            d.symp_prob.append(msim.meta.vals.symp_prob)
            d.mismatch.append(msim.reduce(output=True).compute_fit().mismatch)
            d.infections.append(msim.results['cum_infections'].values[-1])
        df = pd.DataFrame(data=d)

        if 1:#not debug:
            cv.save(f'{resfolder}/{location}_sweeps.df', df)
            sc.saveobj(f'{resfolder}/{location}_sweeps.obj', d)
        sc.toc(T)
