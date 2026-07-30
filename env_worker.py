'''
Explanation for state_dim=52:
TrackErrorDot         = 1  (Normalized distance from center of track, clamped [0,1])
TrackErrorCross       = 1  (Signed lateral offset — left or right of centre)
Velocity              = 3  (XYZ velocity normalized by max_speed)
HeadingAlignmentDot   = 1  (Dot product of kart-front and track-forward in XZ plane)
HeadingAlignmentCross = 1  (Cross product — signed turn direction)
Jumping               = 1
Rotation              = 4  (Quaternion)
Distance              = 1  (Normalized progress along track)
Total per frame       = 13
Frame Stacking x4     = 52
'''


import os
if os.name == 'nt':  # Check if windows
    os.add_dll_directory(r'C:\Program Files\SuperTuxKart 1.5')
import pystk2
import numpy as np
from collections import deque

# Assumed half-width of the track in world units.
# This is used to normalize TrackError to [0, 1].
# Increase this if the kart is frequently hitting values >> 1.
TRACK_HALF_WIDTH = 5.0

# Minimum speed (m/s) before stalling penalty kicks in.
# Set to 2.0 so the kart must maintain a meaningful speed.
STALL_SPEED_THRESHOLD = 2.0


class ProcessState:
    def __init__(self, PathNodes, max_speed=30, map_size=100, track_length=2000):
        self.frame = deque(maxlen=4)    # Window holding last 4 frames
        self.max_speed = max_speed
        self.map_size = map_size
        self.track_length = track_length
        # Use only the center point of each path node segment (index 0 = inner edge, shape [N, 2, 3])
        self.PathNodes = np.array(PathNodes)[:, 0, :].astype(np.float32)

    def processObservation(self, obs):
        KartLocation     = np.array(obs['location'], dtype=np.float32)
        KartFrontLocation = np.array(obs['front'],   dtype=np.float32)

        # --- Track Error (distance from center-line, XZ plane only) ---
        # Ignore Y so that hills don't inflate the error
        nodes_xz = self.PathNodes[:, [0, 2]]          # (N, 2)
        kart_xz  = KartLocation[[0, 2]]               # (2,)
        diff_sq  = np.sum((nodes_xz - kart_xz) ** 2, axis=1)
        AnchorNodeIndex = int(np.argmin(diff_sq))

        raw_error = float(np.sqrt(diff_sq[AnchorNodeIndex]))
        # Normalize and hard-clamp to [0, 1] so off-track spikes don't overwhelm gradients
        TrackErrorDot = np.array(
            [np.clip(raw_error / TRACK_HALF_WIDTH, 0.0, 1.0)],
            dtype=np.float32
        )

        # --- Heading Alignment (XZ plane only) ---
        # Look 5 nodes ahead; wrap around using modulo
        AnchorNode = self.PathNodes[AnchorNodeIndex]
        TargetNode = self.PathNodes[(AnchorNodeIndex + 5) % len(self.PathNodes)]

        # Project onto XZ plane before computing unit vectors
        track_vec_xz = np.array(
            [TargetNode[0] - AnchorNode[0], TargetNode[2] - AnchorNode[2]],
            dtype=np.float32
        )
        front_vec_xz  = np.array(
            [KartFrontLocation[0] - KartLocation[0], KartFrontLocation[2] - KartLocation[2]],
            dtype=np.float32
        )
        offset_vec_xz = kart_xz - AnchorNode[[0, 2]]

        tv_norm = np.linalg.norm(track_vec_xz)
        fv_norm = np.linalg.norm(front_vec_xz)

        if tv_norm > 1e-6 and fv_norm > 1e-6:
            track_vec_xz /= tv_norm
            front_vec_xz /= fv_norm
            heading_dot = float(np.dot(track_vec_xz, front_vec_xz))
            heading_cross = float(track_vec_xz[0] * front_vec_xz[1] - track_vec_xz[1] * front_vec_xz[0])
            track_cross = float(track_vec_xz[0] * offset_vec_xz[1] - track_vec_xz[1] * offset_vec_xz[0])
        else:
            heading_dot = 0.0
            heading_cross = 0.0
            track_cross = 0.0

        HeadingAlignmentDot = np.array([heading_dot], dtype=np.float32)
        HeadingAlignmentCross = np.array([heading_cross], dtype=np.float32)
        TrackErrorCross = np.array([track_cross], dtype=np.float32)

        # --- Velocity (normalized) ---
        vel = np.array(obs['velocity'], dtype=np.float32) / self.max_speed
        vel = np.clip(vel, -1.0, 1.0)

        # --- Other features ---
        jump     = np.array([1.0 if obs['jumping'] else 0.0], dtype=np.float32)
        rotation = np.array(obs['rotation'], dtype=np.float32)
        dist     = np.array(
            [obs.get('distance_down_track', 0.0)],
            dtype=np.float32
        ) / self.track_length

        # Assemble state vector (13 features)
        state = np.concatenate([TrackErrorDot, TrackErrorCross, vel, HeadingAlignmentDot, HeadingAlignmentCross, jump, rotation, dist])

        # Frame stacking: fill deque with copies on first call
        if len(self.frame) == 0:
            for _ in range(4):
                self.frame.append(state)
        else:
            self.frame.append(state)

        return (np.concatenate(self.frame), TrackErrorDot[0], HeadingAlignmentDot[0])


def SingleInstance(rank, pipe):
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

        processor = ProcessState(
            max_speed=30,
            map_size=max_coordinate,
            track_length=track_length,
            PathNodes=track.path_nodes
        )
        RaceEnded = False
        reward    = 0.0

        WorldState.update()
        kart      = WorldState.karts[0]
        prev_dist = kart.distance_down_track

        obs = {
            'location':           kart.location,
            'velocity':           kart.velocity,
            'front':              kart.front,
            'jumping':            kart.jumping,
            'rotation':           kart.rotation,
            'distance_down_track': prev_dist,
        }
        np_obs, TrackErrorDot, HeadingAlignmentDot = processor.processObservation(obs=obs)

        # Send initial observation to model
        pipe.send([np_obs, reward, RaceEnded])

        while True:
            # Receive action from model
            ActionMessage = pipe.recv()

            if ActionMessage == 'TERMINATE':
                return

            action              = pystk2.Action()
            action.steer        = float(ActionMessage[0])
            action.acceleration = float(ActionMessage[1])
            # Brake only if the sigmoid output is clearly positive (threshold 0.5)
            # and acceleration is low — prevents braking and throttle fighting
            action.brake = (ActionMessage[2] > 0.5) and (ActionMessage[1] < 0.3)

            # Step the environment
            RaceEnded = not race.step(action)

            WorldState.update()
            kart         = WorldState.karts[0]
            current_dist = kart.distance_down_track

            obs = {
                'location':           kart.location,
                'velocity':           kart.velocity,
                'front':              kart.front,
                'jumping':            kart.jumping,
                'rotation':           kart.rotation,
                'distance_down_track': current_dist,
            }
            np_obs, TrackErrorDot, HeadingAlignmentDot = processor.processObservation(obs=obs)

            # ---------------------------------------------------------------
            # Reward Calculation
            # ---------------------------------------------------------------
            vel_x, vel_y, vel_z = obs['velocity']
            speed = float((vel_x**2 + vel_y**2 + vel_z**2) ** 0.5)

            delta_dist = current_dist - prev_dist

            # Core progress reward: scaled so one step of good progress ≈ +1
            reward = delta_dist * 80.0

            # Going backward strongly penalized
            if delta_dist < -0.5:
                reward -= 8.0

            # Track centering: weighted by speed so fast+wide is doubly bad
            speed_factor = np.clip(speed / 10.0, 0.0, 1.0)
            reward -= TrackErrorDot * 4.0 * (1.0 + speed_factor)

            # Heading bonus: reward for facing the right way
            reward += HeadingAlignmentDot * 2.0

            # Stalling penalty (must maintain a meaningful speed)
            if speed < STALL_SPEED_THRESHOLD:
                reward -= 3.0

            # Backward-distance penalty (went past start)
            if current_dist < 0:
                reward -= 5.0

            # ---- Terminal rewards ----
            if RaceEnded:
                if current_dist >= track_length * 0.99:
                    # Finished the race! Large completion bonus
                    reward += 500.0
                else:
                    # Did not finish (timeout / crash)
                    reward -= 20.0

            prev_dist = current_dist

            # Send observation back to model
            pipe.send([np_obs, float(reward), RaceEnded])

    finally:
        # Critical Cleanup
        race.stop()
        del race
        pystk2.clean()
