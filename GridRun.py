from Persistence import Persistence
import logging

def construct_configs(grid):
    configs = [{}]
    for key, values in grid.items():
        if type(values) != list:
            values = [values]
        new_configs = []
        for value in values:
            new_configs += [{**con, key: value} for con in configs]
        configs = new_configs
    return configs

def grid_run(run_experiment_fnc, grid, path=None, ignore_previous_results=False):
    logger = logging.getLogger('grid_run')
    configs = construct_configs(grid)
    logger.info('Starting experiment with grid %s (%d runs)', grid, len(configs))
    
    # Main loop.
    for config_num, config in enumerate(configs):

        # Check whether it has been done before.
        if path is not None:
            with Persistence(path) as db:
                len_ids = len(db.get_persistence_ids(config))
                if len_ids > 0:
                    logger.info('Already done (%d time(s)): %s', len_ids, config)
                    if ignore_previous_results:
                        logger.info('Repeating experiment')    
                    else:
                        logger.info('Skipping experiment')
                        continue
        
        # Big log.
        log_str = 'RUN CONFIG ({}/{})'.format(config_num + 1, len(configs))
        log_str = '\n' + '-' * len(log_str) + '\n' + log_str + '\n' + '-' * len(log_str) + '\n'
        log_str += '\n'.join([key + ': ' + str(value) for key, value in config.items()])
        log_str += '\n----------'
        logger.info(log_str)
        
        # Run experiment.
        experiment_results = run_experiment_fnc(**config, config=config)
        
        # Log the end of the experiment. Print results if found.
        if type(experiment_results) == float:
            logger.info('Run finished. Result: %.4f', experiment_results)
        elif type(experiment_results) == dict and 'test_score' in experiment_results:
            logger.info('Run finished. Test score: %.4f', experiment_results['test_score'])
        else:
            logger.info('Run finished.')
        
        # Store results.
        if path is not None:
            with Persistence(path) as db:
                db.add_new_entry(config, experiment_results)
            logger.info('Results added to persistence in %s.', path)

if __name__ == '__main__':
    grid = {
        'a': 1,
        'b': [2, 3, 4],
        'c': ['Hello', 'world!']
        }
    
    def run_experiment(a, b, c):
        print(a, b, c)
    
    logging.basicConfig(level=logging.INFO)
    
    grid_run(run_experiment, grid)
    
    
    
