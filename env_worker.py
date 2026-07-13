'''
Explanation for state_dim=60:
Location = 3
Velocity = 3
Front = 3
Jumping = 1
Rotation = 4
Distance = 1
Total = 15
Frame Stacking (x4) = 60
'''

import os
if os.name == 'nt': #Check if windows
    os.add_dll_directory(r'C:\Program Files\SuperTuxKart 1.5')
import pystk2
import numpy as np
from collections import deque

class ProcessState:
	def __init__(self, max_speed=30, map_size=100, track_length=2000):
		self.frame = deque(maxlen=4)    # Window holding last 4 frames
		self.max_speed = max_speed
		self.map_size = map_size
		self.track_length = track_length

	def processObservation(self, obs):
		loc = np.array(obs["location"], dtype=np.float32) / self.map_size
		loc = np.clip(loc, -1.0, 1.0)

		vel = np.array(obs["velocity"], dtype=np.float32) / self.max_speed
		vel = np.clip(vel, -1.0, 1.0)

		front = np.array(obs["front"], dtype=np.float32) / self.map_size
		front = np.clip(front, -1.0, 1.0)

		jump = np.array([1.0 if obs["jumping"] else 0.0], dtype=np.float32)

		rotation = np.array(obs["rotation"], dtype=np.float32)

		dist = np.array([obs.get("distance_down_track", 0.0)], dtype=np.float32) / self.track_length

		state = np.concatenate([loc, vel, front, jump, rotation, dist])

		if len(self.frame) == 0:
			for _ in range(4):
				self.frame.append(state)
		else:
			self.frame.append(state)

		return np.concatenate(self.frame)

def SingleInstance(rank,pipe):
	pystk2.init(pystk2.GraphicsConfig.none())
	WorldState = pystk2.WorldState()
	config = pystk2.RaceConfig(track='lighthouse', num_kart=1, laps=1)
	config.players[0].controller = pystk2.PlayerConfig.Controller.PLAYER_CONTROL
	race = pystk2.Race(config)
	try:
		race.start()
			
		# Track details must be loaded after race start
		track = pystk2.Track()
		track.update()
		track_length = track.length
		max_coordinate = np.max(np.abs(track.path_nodes))
		
		processor = ProcessState(max_speed=30, map_size=max_coordinate, track_length=track_length)
		RaceEnded = False
		reward = 0.0

		WorldState.update()
			
		kart = WorldState.karts[0]
		prev_dist = kart.distance_down_track
		
		obs = {
			"location": kart.location,
			"velocity": kart.velocity,
			"front": kart.front,
			"jumping": kart.jumping,
			"rotation": kart.rotation,
			"distance_down_track": prev_dist
		}
		np_obs = processor.processObservation(obs=obs)
		#Send observation to model
		pipe.send([np_obs,reward,RaceEnded])

		while True:
			#Receive action from model
			ActionMessage = pipe.recv()

			if ActionMessage == 'TERMINATE':
				return

			action = pystk2.Action()
			action.steer = ActionMessage[0]
			action.acceleration = ActionMessage[1]
			action.brake = True if ActionMessage[2] > 0.5 else False

			# Step the environment
			RaceEnded = not race.step(action)

			WorldState.update()

			kart = WorldState.karts[0]
			current_dist = kart.distance_down_track
			
			obs = {
				"location": kart.location,
				"velocity": kart.velocity,
				"front": kart.front,
				"jumping": kart.jumping,
				"rotation": kart.rotation,
				"distance_down_track": current_dist
			}
			np_obs = processor.processObservation(obs=obs)

			# Reward Calculation
			vel_x, vel_y, vel_z = obs['velocity']
			speed = (vel_x**2 + vel_y**2 + vel_z**2)**0.5
			
			delta_dist = current_dist - prev_dist
			reward = delta_dist * 100.0
			
			if speed < 1.0:
				reward -= 5.0
				
			if current_dist < 0:
				reward -= 10.0
				
			prev_dist = current_dist
				
			#Send observation to model
			pipe.send([np_obs,reward,RaceEnded])

	finally:
		# Critical Cleanup
		race.stop()
		del race
		pystk2.clean()
