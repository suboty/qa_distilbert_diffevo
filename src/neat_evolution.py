"""
A parallel version of XOR using neat.parallel.

Since XOR is a simple experiment, a parallel version probably won't run any
faster than the single-process version, due to the overhead of
inter-process communication.

If your evaluation function is what's taking up most of your processing time
(and you should check by using a profiler while running single-process),
you should see a significant performance improvement by evaluating in parallel.

This example is only intended to show how to do a parallel experiment
in neat-python.  You can of course roll your own parallelism mechanism
or inherit from ParallelEvaluator if you need to do something more complicated.
"""

import multiprocessing
import os

import neat

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


class CNEAT:
    def __init__(self,
                 config_path=os.path.join(os.path.dirname(__file__),
                                          'config-feedforward')):
        self.config_path = config_path
        self.eval_genome = xor_eval_genome
        self.evalute = xor_evalute_winner

    def set_eval_genome(self, func):
        self.eval_genome = func

    def set_evalutte(self, func):
        self.evalute = func

    def run(self, config_file):
        # Load configuration.
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        # Run for up to 300 generations.
        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), xor_eval_genome)
        winner = p.run(pe.evaluate, 300)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

        return winner_net

    def __call__(self, *args, **kwargs):
        best_net = self.run(self.config_path)
        self.evalute(best_net)


def xor_eval_genome(genome, config):
    """
    This function will be run in parallel by ParallelEvaluator.  It takes two
    arguments (a single genome and the genome class configuration data) and
    should return one float (that genome's fitness).

    Note that this function needs to be in module scope for multiprocessing.Pool
    (which is what ParallelEvaluator uses) to find it.  Because of this, make
    sure you check for __main__ before executing any code (as we do here in the
    last few lines in the file), otherwise you'll have made a fork bomb
    instead of a neuroevolution demo. :)
    """

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    error = 4.0
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        error -= (output[0] - xo[0]) ** 2
    return error


def xor_evalute_winner(winner_net):
    print('\nOutput:')
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
