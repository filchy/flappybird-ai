import neat


def fitness_function():
	pass

def neat_net(conf_path, function):
	# Load config
	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
								neat.DefaultSpeciesSet, neat.DefaultStagnation, 
								conf_path)
	# Creating population
	p = neat.Population(config)

	# Create reporter to show progress in the terminal
	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)

	winner = p.run(function, 50)