from bird_object import Bird
from pipe_object import Pipe
from base_object import Base

from neat_network import neat_net

import pygame
import os
import random
import neat


WIN_HEIGHT = 800
WIN_WIDTH = 500

BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

def draw_window(win, birds, pipes, base):
	win.blit(BG_IMG, (0,0))

	for pipe in pipes:
		pipe.draw(win)

	base.draw(win)

	for bird in birds:
		bird.draw(win)
	pygame.display.update()

def eval_genomes(genomes, conf):
	score = 0
	nets = []
	ge = []
	birds = []

	for _,g in genomes:
		net = neat.nn.FeedForwardNetwork.create(g, conf)
		nets.append(net)
		birds.append(Bird(230,350))
		g.fitness = 0
		ge.append(g)

	base = Base(730)
	pipes = [Pipe(500)]

	win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
	clock = pygame.time.Clock()

	run = True
	while run:
		clock.tick(30)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				pygame.quit()
				quit()
				break

		pipe_ind = 0
		if len(birds) > 0:
			if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
				pipe_ind = 1
		else:
			run = False
			break

		for x, bird in enumerate(birds):
			bird.move()
			ge[x].fitness += 0.1

			output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

			if output[0] > .5:
				bird.jump()

		#bird.move()
		add_pipe = False
		rem = []
		# collisions
		for pipe in pipes:
			for x, bird in enumerate(birds):
				if pipe.collision(bird):
					ge[x].fitness -= 1
					birds.remove(bird)

				if not pipe.passed and pipe.x < bird.x:
					pipe.passed = True
					add_pipe = True
			
			if pipe.x + pipe.PIPE_TOP.get_width() < 0:
				rem.append(pipe)

			pipe.move()

		if add_pipe:
			score += 1
			for g in ge:
				g.fitness += 5
			pipes.append(Pipe(500))

		for r in rem:
			pipes.remove(r)

		for x, bird in enumerate(birds):
			if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
				birds.pop(x)
				nets.pop(x)
				ge.pop(x)

		base.move()
		draw_window(win, birds, pipes, base)



if __name__ == '__main__':
	local_dir = os.path.dirname(__file__)
	cong_path = os.path.join(local_dir, "config-feedforward.txt")
	neat_net(cong_path, eval_genomes)
