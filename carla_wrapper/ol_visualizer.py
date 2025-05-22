import carla
import pygame
import numpy as np
import weakref
from queue import Queue

class Visualizer:
    def __init__(self, world, display_width=800, display_height=600):
        self.world = world
        self.display_width = display_width
        self.display_height = display_height
        self.rgb_queue = Queue(maxsize=5)
        self.rgb_array = None

        # Init PyGame safely
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.display_width, self.display_height),
            pygame.SWSURFACE
        )
        pygame.display.set_caption("CARLA Visualizer")

        mono = pygame.font.match_font('ubuntumono') or pygame.font.get_default_font()
        self.font = pygame.font.Font(mono, 14)

    def set_camera_listener(self, camera, camera_type='rgb'):
        weak_self = weakref.ref(self)

        def callback(image):
            self_ref = weak_self()
            if not self_ref:
                return
            try:
                self_ref.rgb_queue.put(image.raw_data, timeout=0.01)
                print(f"[Visualizer] Frame {image.frame} enqueued.")
            except:
                print("[Visualizer] Frame dropped (queue full)")

        camera.listen(callback)

    def render(self, throttle, steer, brake, timestamp):
        # Pull most recent image if available
        while not self.rgb_queue.empty():
            try:
                raw = self.rgb_queue.get_nowait()
                array = np.frombuffer(raw, dtype=np.uint8).copy()
                array = array.reshape((self.display_height, self.display_width, 4))[:, :, :3]
                self.rgb_array = array[:, :, ::-1]
                print("[Visualizer] Frame updated.")
            except Exception as e:
                print(f"[Visualizer] Error processing frame: {e}")

        if self.rgb_array is not None:
            try:
                surface = pygame.surfarray.make_surface(self.rgb_array.swapaxes(0, 1))
                self.display.blit(surface, (0, 0))
            except Exception as e:
                print(f"[Visualizer] surface conversion failed: {e}")

        self.render_text(throttle, steer, brake, timestamp)
        pygame.display.flip()

    def render_text(self, throttle, steer, brake, timestamp):
        info_text = [
            f"Timestamp: {timestamp:.2f}",
            f"Throttle : {throttle:.2f}",
            f"Steer    : {steer:.2f}",
            f"Brake    : {brake:.2f}"
        ]

        info_surface = pygame.Surface((220, self.display_height // 4))
        info_surface.set_alpha(100)
        self.display.blit(info_surface, (0, 0))

        v_offset = 10
        for line in info_text:
            surface = self.font.render(line, True, (255, 255, 255))
            self.display.blit(surface, (8, v_offset))
            v_offset += 18

    def close(self):
        pygame.quit()
