'''
Explanation for state_dim=44:
TrackError = 1 (Distance from center of track)
Velocity = 3
HeadingAlignment = 1 (Dot product of front vector and track vector)
Jumping = 1
Rotation = 4
Distance = 1
Total = 11
Frame Stacking (x4) = 44
'''

import os
if os.name == 'nt': #Check if windows
    os.add_dll_directory(r'C:\Program Files\SuperTuxKart 1.5')
import pystk2
import numpy as np
from collections import deque

class ProcessState:
	def __init__(self, PathNodes, max_speed=30, map_size=100, track_length=2000):
		self.frame = deque(maxlen=4)    # Window holding last 4 frames
		self.max_speed = max_speed
		self.map_size = map_size
		self.track_length = track_length
		self.PathNodes = np.array(PathNodes)[:,0,:]

	def processObservation(self, obs):
		KartLocation = np.array(obs['location'], dtype=np.float32)
		KartFrontLocation = np.array(obs['front'], dtype=np.float32)
		SquaredNodes = np.sum((self.PathNodes-KartLocation)**2, axis=1)
		AnchorNodeIndex = np.argmin(SquaredNodes)
		AnchorNode = self.PathNodes[AnchorNodeIndex]
		TargetNode = self.PathNodes[(AnchorNodeIndex+5)%len(self.PathNodes)]

		TrackError = np.array([np.sqrt(SquaredNodes[AnchorNodeIndex])/10], dtype=np.float32) #10 is what I assume the max track width to be, I'll change it when required

		TargetVector = (TargetNode-AnchorNode)/np.linalg.norm(TargetNode-AnchorNode) #Divide with magnitude to obtain direction unit vector
		FrontVector = (KartFrontLocation-KartLocation)/np.linalg.norm(KartFrontLocation-KartLocation)
		HeadingAlignment = np.array([np.dot(TargetVector,FrontVector)], dtype=np.float32)

		vel = np.array(obs["velocity"], dtype=np.float32) / self.max_speed
		vel = np.clip(vel, -1.0, 1.0)

		jump = np.array([1.0 if obs["jumping"] else 0.0], dtype=np.float32)

		rotation = np.array(obs["rotation"], dtype=np.float32)

		dist = np.array([obs.get("distance_down_track", 0.0)], dtype=np.float32) / self.track_length

		state = np.concatenate([TrackError, vel, HeadingAlignment, jump, rotation, dist])

		if len(self.frame) == 0:
			for _ in range(4):
				self.frame.append(state)
		else:
			self.frame.append(state)

		return (np.concatenate(self.frame),TrackError[0],HeadingAlignment[0])

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
		
		processor = ProcessState(max_speed=30, map_size=max_coordinate, track_length=track_length, PathNodes=track.path_nodes)
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
		np_obs,TrackError,HeadingAlignment = processor.processObservation(obs=obs)
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
			np_obs,TrackError,HeadingAlignment = processor.processObservation(obs=obs)

			# Reward Calculation
			vel_x, vel_y, vel_z = obs['velocity']
			speed = (vel_x**2 + vel_y**2 + vel_z**2)**0.5
			
			delta_dist = current_dist - prev_dist
			reward = delta_dist * 100.0
			reward -= (TrackError * 5.0)
			reward += (HeadingAlignment * 1.5)
			
			if speed < 1.0:
				reward -= 5.0
				
			if current_dist < 0:
				reward -= 10.0
				
			prev_dist = current_dist
				
			#Send observation to model
			pipe.send([np_obs,float(reward),RaceEnded])

	finally:
		# Critical Cleanup
		race.stop()
		del race
		pystk2.clean()
