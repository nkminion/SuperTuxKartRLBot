import os
if os.name == 'nt':
    os.add_dll_directory(r'C:\Program Files\SuperTuxKart 1.5')
import pystk2
import numpy as np
from collections import deque

TRACK_HALF_WIDTH = 5.0
STALL_SPEED_THRESHOLD = 2.0


class ProcessState:
    def __init__(self, PathNodes, max_speed=30, map_size=100, track_length=2000):
        self.frame = deque(maxlen=4)
        self.max_speed = max_speed
        self.map_size = map_size
        self.track_length = track_length
        self.PathNodes = np.array(PathNodes)[:, 0, :].astype(np.float32)

    def processObservation(self, obs):
        KartLocation     = np.array(obs['location'], dtype=np.float32)
        KartFrontLocation = np.array(obs['front'],   dtype=np.float32)

        nodes_xz = self.PathNodes[:, [0, 2]]
        kart_xz  = KartLocation[[0, 2]]
        diff_sq  = np.sum((nodes_xz - kart_xz) ** 2, axis=1)
        AnchorNodeIndex = int(np.argmin(diff_sq))

        raw_error = float(np.sqrt(diff_sq[AnchorNodeIndex]))
        TrackErrorDot = np.array(
            [np.clip(raw_error / TRACK_HALF_WIDTH, 0.0, 1.0)],
            dtype=np.float32
        )

        AnchorNode = self.PathNodes[AnchorNodeIndex]
        TargetNode = self.PathNodes[(AnchorNodeIndex + 5) % len(self.PathNodes)]

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
        TrackErrorCross = np.array([np.clip(track_cross / TRACK_HALF_WIDTH, -1.0, 1.0)], dtype=np.float32)

        vel = np.array(obs['velocity'], dtype=np.float32) / self.max_speed
        vel = np.clip(vel, -1.0, 1.0)

        jump     = np.array([1.0 if obs['jumping'] else 0.0], dtype=np.float32)
        rotation = np.array(obs['rotation'], dtype=np.float32)
        dist     = np.array(
            [obs.get('distance_down_track', 0.0)],
            dtype=np.float32
        ) / self.track_length

        state = np.concatenate([TrackErrorDot, TrackErrorCross, vel, HeadingAlignmentDot, HeadingAlignmentCross, jump, rotation, dist])

        if len(self.frame) == 0:
            for _ in range(4):
                self.frame.append(state)
        else:
            self.frame.append(state)

        return (np.concatenate(self.frame), TrackErrorDot[0], HeadingAlignmentDot[0])


def SingleInstance(rank, pipe, hd=False, frame_delay=0.0):
    race = None
    try:
        if hd:
            pystk2.init(pystk2.GraphicsConfig.hd())
            if frame_delay <= 0:
                frame_delay = 0.02
        else:
            pystk2.init(pystk2.GraphicsConfig.none())
        WorldState = pystk2.WorldState()
        config = pystk2.RaceConfig(track='lighthouse', num_kart=1, laps=1)
        config.players[0].controller = pystk2.PlayerConfig.Controller.PLAYER_CONTROL
        race = pystk2.Race(config)
        race.start()

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
        StuckFrameCounter = 0

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

        pipe.send([np_obs, reward, RaceEnded])

        while True:
            ActionMessage = pipe.recv()

            if ActionMessage == 'TERMINATE':
                return

            action              = pystk2.Action()
            action.steer        = float(ActionMessage[0])
            action.acceleration = float(ActionMessage[1])
            action.brake        = (ActionMessage[2] > 0.5) and (ActionMessage[1] < 0.3)

            RaceEnded = not race.step(action)

            if frame_delay > 0:
                import time
                time.sleep(frame_delay)

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

            vel_x, vel_y, vel_z = obs['velocity']
            speed = float((vel_x**2 + vel_y**2 + vel_z**2) ** 0.5)

            delta_dist = current_dist - prev_dist
            
            # Wrap-around fix
            if delta_dist > track_length / 2.0:
                delta_dist -= track_length
            elif delta_dist < -track_length / 2.0:
                delta_dist += track_length

            reward = delta_dist * 80.0

            if delta_dist < -0.5:
                reward -= 8.0

            speed_factor = np.clip(speed / 10.0, 0.0, 1.0)
            reward -= TrackErrorDot * 4.0 * (1.0 + speed_factor)

            reward += HeadingAlignmentDot * 2.0

            if speed < STALL_SPEED_THRESHOLD:
                reward -= 3.0

            if current_dist < 0:
                reward -= 5.0

            if speed < STALL_SPEED_THRESHOLD:
                StuckFrameCounter += 1
            else:
                StuckFrameCounter = 0

            if not hd and StuckFrameCounter > 60:
                reward    = -200.0
                RaceEnded = True

            if RaceEnded and StuckFrameCounter <= 60:
                if current_dist >= track_length * 0.99:
                    reward += 500.0
                else:
                    reward -= 20.0

            prev_dist = current_dist

            pipe.send([np_obs, float(reward), RaceEnded])

            if RaceEnded:
                if hd:
                    import time
                    time.sleep(2.0)
                return

    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        if race is not None:
            try:
                race.stop()
            except Exception:
                pass
            del race
        try:
            pystk2.clean()
        except Exception:
            pass


def SingleInstanceHD(rank, pipe, frame_delay=0.02):
    SingleInstance(rank, pipe, hd=True, frame_delay=frame_delay)
