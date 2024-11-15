import carla
import pygame
import numpy as np
import weakref

DEFAULT_VIS_TIME = 30000.0

class Visualizer:
    def __init__(self, world, display_width=800, display_height=600):
        self.world = world
        self.display_width = display_width
        self.display_height = display_height
        self.rgb_surface = None
        self.depth_surface = None

        # Initialize pygame for rendering
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.display_width * 2, self.display_height),  # Double width to display RGB and depth side by side
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("CARLA Visualizer")

        # Set font for text display
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self.font = pygame.font.Font(mono, 14)

    def set_camera_listener(self, camera, camera_type='rgb'):
        # Attach a listener to the camera to get frames in real time
        weak_self = weakref.ref(self)
        camera.listen(lambda image: Visualizer._parse_image(weak_self, image, camera_type))

    @staticmethod
    def _parse_image(weak_self, image, camera_type):
        self = weak_self()
        if not self:
            return

        # Convert CARLA image to Pygame surface
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))

        if camera_type == 'rgb':
            array = array[:, :, :3]  # Keep only RGB channels
            array = array[:, :, ::-1]  # Convert from BGRA to RGB
            self.rgb_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        elif camera_type == 'depth':
            # Convert depth to grayscale 8-bit format for display
            depth_array = (array[:, :, 0] / 255.0 * 255).astype(np.uint8)  # Normalize depth values
            depth_rgb = np.stack((depth_array,)*3, axis=-1)  # Expand to 3 channels
            self.depth_surface = pygame.surfarray.make_surface(depth_rgb.swapaxes(0, 1))

    def render(self, throttle, steer, brake, timestamp):
        # Display the RGB and depth camera feeds side by side
        if self.rgb_surface:
            self.display.blit(self.rgb_surface, (0, 0))
        if self.depth_surface:
            self.display.blit(self.depth_surface, (self.display_width, 0))  # Display depth feed next to RGB

        # Overlay text info
        self.render_text(throttle, steer, brake, timestamp)
        pygame.display.flip()

    def render_text(self, throttle, steer, brake, timestamp):
        # Information overlay on the camera feed
        info_text = [
            "Timestamp: {}".format(timestamp),
            "Throttle : {:.2f}".format(throttle),
            "Steer    : {:.2f}".format(steer),
            "Brake    : {:.2f}".format(brake),
        ]

        # Display the information box on the image
        info_surface = pygame.Surface((220, self.display_height // 4))
        info_surface.set_alpha(100)
        self.display.blit(info_surface, (0, 0))

        # Render the text
        v_offset = 10
        for line in info_text:
            surface = self.font.render(line, True, (255, 255, 255))
            self.display.blit(surface, (8, v_offset))
            v_offset += 18

    def close(self):
        pygame.quit()
