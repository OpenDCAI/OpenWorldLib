"""Utilities for recording simulation videos and logging metadata."""

import csv
import os
import random

import cv2


class VideoRecorder:
    """Record frames from a humanoid-mounted camera and export video/CSV logs."""

    def __init__(
            self,
            communicator,
            humanoid,
            output_dir='.',
            video_name='output.mp4',
            resolution=(1440, 720),
            fps=25.0,
            frame_num=500,
            move_pattern=None,
            camera_mode='lit'):
        """Initialize the recorder with target communicator and humanoid.

        Parameters
        ----------
        communicator: object
            Game communicator providing camera and control APIs.
        humanoid: object
            Humanoid agent that owns the camera to record from.
        output_dir: str
            Directory to write the output files (video and CSV logs).
        video_name: str
            Output video filename.
        resolution: tuple[int, int]
            Target camera resolution (width, height).
        fps: float
            Frames per second for the output video.
        frame_num: int
            Total number of frames to record. Use >0 to enable progress.
        camera_mode: str
            Unreal render mode, e.g., 'lit', 'depth', etc.
        """
        self.communicator = communicator
        self.humanoid = humanoid
        self.output_dir = output_dir
        self.video_path = os.path.join(output_dir, video_name)
        self.csv_camera = os.path.join(output_dir, 'camera_position.csv')
        self.csv_action = os.path.join(output_dir, 'humanoid_action.csv')
        self.resolution = resolution
        self.fps = fps
        self.frame_num = frame_num
        self.camera_mode = camera_mode
        if move_pattern:
            self.move_pattern = lambda timestamp: move_pattern(self, timestamp)
        else:
            self.move_pattern = self._move_pattern

        # set resolution
        self.communicator.unrealcv.set_camera_resolution(
            self.humanoid.camera_id, resolution
        )

        os.makedirs(output_dir, exist_ok=True)

    def record(self):
        """Run the recording loop and persist video and CSV logs.

        Returns
        -------
        str
            Path to the saved video file.
        """
        images = []
        camera_position = []
        humanoid_actions = []

        timestamp = 0
        stop = False

        while not stop:
            # get image
            try:
                img = self.communicator.get_camera_observation(
                    self.humanoid.camera_id, self.camera_mode, mode='direct'
                )
            except Exception as e:
                print('❌ Error getting image:', e)
                break

            # get camera state
            camera_loc = self.communicator.unrealcv.get_camera_location(self.humanoid.camera_id)
            camera_rot = self.communicator.unrealcv.get_camera_rotation(self.humanoid.camera_id)

            images.append(img)
            camera_position.append([timestamp, camera_loc, camera_rot])

            # log action
            actions = self.move_pattern(timestamp)
            humanoid_actions.extend(actions)

            # progress bar
            timestamp += 1
            if self.frame_num > 0:
                progress = timestamp / self.frame_num
                bar_length = 30
                filled = int(bar_length * progress)
            bar = '#' * filled + '-' * (bar_length - filled)
            print(f'\rProgress: |{bar}| {timestamp}/{self.frame_num} ({progress:.1%})', end='')

            if len(images) >= self.frame_num:
                stop = True

            # Tick
            self.communicator.unrealcv.tick()

        print('\n✅ Recording finished.')
        self.save_video(images)
        self.save_csv(camera_position, humanoid_actions)
        return self.video_path

    def _move_pattern(self, timestamp):
        """Generate a movement pattern for the humanoid at given timestamp."""
        actions = []
        step = timestamp

        # --- move forward ---
        self._apply_action('move_forward', None, timestamp, actions)

        # --- big turn ---
        if step % random.randint(30, 50) == 0:
            turn_angle = random.randint(60, 160)
            turn_dir = random.choice(['left', 'right'])
            action = f'rotate_{turn_dir}'
            self._apply_action(action, turn_angle, timestamp, actions)

        # --- small jitter ---
        elif step % random.randint(20, 40) == 0:
            jitter_angle = random.randint(-10, 10)
            if jitter_angle > 0:
                self._apply_action('rotate_right', jitter_angle, timestamp, actions)
            elif jitter_angle < 0:
                self._apply_action('rotate_left', abs(jitter_angle), timestamp, actions)

        return actions

    def _apply_action(self, action, value, timestamp, actions):
        """Apply a single action and log it into the actions list."""
        if action == 'move_forward':
            self.communicator.humanoid_move_forward(self.humanoid.id)
            actions.append([timestamp, 'move_forward'])
        elif action == 'rotate_right':
            self.communicator.humanoid_rotate(self.humanoid.id, value, 'right')
            actions.append([timestamp, 'rotate_right'])
        elif action == 'rotate_left':
            self.communicator.humanoid_rotate(self.humanoid.id, value, 'left')
            actions.append([timestamp, 'rotate_left'])

    def save_video(self, images):
        """Save frames to an MP4 file using OpenCV."""
        if not images:
            print('⚠️ No frames to save.')
            return
        h, w, _ = images[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, self.fps, (w, h))
        for img in images:
            out.write(img)  # img must be in BGR format
        out.release()
        print(f'💾 Video saved to {self.video_path}')

    def save_csv(self, camera_position, humanoid_actions):
        """Save camera pose timeline and humanoid actions into CSV files."""
        with open(self.csv_camera, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'camera_loc', 'camera_rot'])
            for loc in camera_position:
                writer.writerow(loc)
        print(f'💾 Camera positions saved to {self.csv_camera}')

        with open(self.csv_action, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'action'])
            for action in humanoid_actions:
                writer.writerow(action)
        print(f'💾 Actions saved to {self.csv_action}')
