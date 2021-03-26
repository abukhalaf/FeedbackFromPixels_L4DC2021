#!/usr/bin/env python

# Author: Murad Abu-Khalaf, MIT CSAIL.

"""
    This runs a closed-loop test of the view synthesizer based controller.

    It requires CARLA simulator to be running.

"""

import glob
import os
import sys
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time
import weakref
import matplotlib.pyplot as plt
import cv2

try:
    import pygame
except ImportError:
    raise RuntimeError('pygame import error')

try:
    import trainViewSynthesizerNNet
except  ImportError:
    raise RuntimeError('trainViewSynthesizerNNet import error')

import torch
import torch.nn as nn
import torch.nn.functional as F

class CarFollowing:
    """
    Spawns two vehciles, a leader and a follower.

    A forward looking camera is attached to the follower car.
    """

    def __init__(self):        
        self.actor_list = []
        self.surface = None
        self.distance = np.empty((0,5))
        self.row = np.empty((0,5))
        self.time = None
        self.camera = None
        self.NN = None
        self.img = None
        self.PGain = None
        self.img_ref = None
        self.img_ref0 = None

        try:
            net = trainViewSynthesizerNNet.Net()
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                print("Running on GPU")
            else:
                device = torch.device("cpu")
                print("Running on CPU")
                torch.cuda.device_count()
            net.to(device)

            net.load_state_dict(torch.load('viewSynthesizerNNet.pth'))
            net.eval()
            self.NN = net
            print(net)
        except Exception:
            raise RuntimeError('cannot load neural network')

        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(2.0)
            world = client.get_world()
            self.world = world

            # Stop weather conditions
            self.world.set_weather(carla.WeatherParameters(
                cloudiness = 0.0, wind_intensity = 0.0,
                precipitation = 0.0, precipitation_deposits = 0.0,
                sun_azimuth_angle = 0.0, sun_altitude_angle = 90.0 ))

            # Specify cars model and color
            blueprint_library = world.get_blueprint_library()
            follower_bp = random.choice(blueprint_library.filter('prius'))
            leader_bp = random.choice(blueprint_library.filter('impala'))

            if follower_bp.has_attribute('color'):
                color = random.choice(follower_bp.get_attribute('color').recommended_values)
                follower_bp.set_attribute('color', '0,0,255')

            if leader_bp.has_attribute('color'):
                color = random.choice(leader_bp.get_attribute('color').recommended_values)
                leader_bp.set_attribute('color', '255,0,0')

            # Town03: Follower (blue): starts @ x = -10.0, y = 40.0, Leader (red) starts @ x = -10, y = 50.0
            follower_transform = carla.Transform(carla.Location(x = -10, y = 40, z=2.5),
                carla.Rotation(pitch = 0, yaw = 90, roll = 0))
            leader_transform = carla.Transform(carla.Location(x = -10, y = 50, z=2.5),
                carla.Rotation(pitch = 0, yaw = 90, roll = 0))
            self.follower = world.try_spawn_actor(follower_bp, follower_transform)
            self.leader = world.try_spawn_actor(leader_bp, leader_transform)

            # Storing created actors so we may destroy later.
            self.actor_list.append(self.follower)
            print('created %s' % self.follower.type_id)
            self.actor_list.append(self.leader)
            print('created %s' % self.leader.type_id)

            # Attach a camera to the follower
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('gamma', str(2.2))
            camera_bp.set_attribute('sensor_tick','0')
            camera_transform = carla.Transform(carla.Location(x=0.0, z=2.4))
            self.camera = world.spawn_actor(camera_bp, camera_transform, attach_to=self.follower)
            self.actor_list.append(self.camera)
            print('created %s' % self.camera.type_id)
            print(self.camera.attributes['sensor_tick'])
            weak_self = weakref.ref(self)
            self.camera.listen(lambda image: CarFollowing.sensorCallback(weak_self, image))

        finally:
            print("....................................")

    @staticmethod
    def sensorCallback(weak_self, image):
        """
        Camera callback

        Updates front camera feed and records speeds and spacing.
        """
        self = weak_self()
        
        # Update the camera feed
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]   # BGRA ==> #BGR
        array = array[:, :, ::-1] # BGR ==> RGB
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        # Store the camera view to be accessed by controller method
        img = cv2.resize(array, (128, 128))
        self.img = img.transpose(2,0,1) # HxWxC ==> CxHxW

        # Record speeds and spacing between cars
        loc = self.leader.get_location()
        v2 = np.sqrt((self.leader.get_velocity().x)**2 + (self.leader.get_velocity().y)**2)
        v1 = np.sqrt((self.follower.get_velocity().x)**2 + (self.follower.get_velocity().y)**2)
        d = np.sqrt((image.transform.location.x - loc.x)**2 + (image.transform.location.y - loc.y)**2)
        self.row = np.array([image.timestamp, d, v1, v2, 0])
        #print(self.row)
        return

    def PControl(self, img_err):
        if self.PGain is None:
            K = np.ones((3, 128, 128)) / 50.0
            self.PGain = K

        K = self.PGain
        f = np.sum(K * img_err )
        f_bar = 0 
        f_tilda = f_bar - f
        u = f_tilda        
        #print(u)
        #u = np.clip(u, -0.5, 0.5)
        #print(u)

        controlValue = carla.VehicleControl(throttle = 0.2,
                                            steer = 0,
                                            brake = 0,
                                            hand_brake = False,
                                            reverse = False,
                                            manual_gear_shift = False,
                                            gear = 1)
        self.leader.apply_control(controlValue)

        if u<0:        
            controlValue = carla.VehicleControl(throttle=np.absolute(u),
                                                steer=0,
                                                brake=0,
                                                hand_brake=False,
                                                reverse=True,
                                                manual_gear_shift=False,
                                                gear=1)
        else:
            controlValue = carla.VehicleControl(throttle=np.absolute(u),
                                                steer=0,
                                                brake=0,
                                                hand_brake=False,
                                                reverse=False,
                                                manual_gear_shift=False,
                                                gear=1)
        self.follower.apply_control(controlValue)
        self.row[4] = u
        self.distance = np.concatenate((self.distance, [self.row]), axis=0)        
        return u

    def getErrorSignal_BlockDiagram1(self, yref):
        """
        Implements Block Diagram 1 in L4DC paper.
        """
        #if self.img_ref is None:
        net = self.NN
        #print(self.img)
        camera_feed = torch.Tensor([self.img]).to('cuda:0)')
        img_ref = net(camera_feed,torch.Tensor([[[[yref]]]]).to('cuda:0)'))
        img_ref = img_ref.to('cpu')
        self.img_ref = img_ref.detach().numpy()[0]
        #plt.figure(1)
        #plt.show(block = False)
        #plt.imshow(self.img_ref.transpose(1, 2, 0)/255.0, cmap="viridis")
        #plt.pause(0.001)
        #plt.draw()
        
        #print(self.img_ref)
        #print(self.img)
        img_err = (self.img_ref - self.img)/255.0
        #print(img_err)
        return img_err

    def getErrorSignal_BlockDiagram2(self, yref):
        """
        Implements Block Diagram 2 in L4DC paper.
        """

        # Note this implementation is not using yet a neural network
        # to correctly keep the background only. To approximate this
        # using currently trained network, we generate a reference 
        # for as far as possible which is 30m. With that, it currently 
        # partially works if K = np.ones((3, 128, 128)) / 372.0, when
        # desired spacing is 10 and throttle of leader is 0.16.
        net = self.NN

        camera_feed = torch.Tensor([self.img]).to('cuda:0)')
        img_ref = net(camera_feed,torch.Tensor([[[[yref]]]]).to('cuda:0)'))
        img_ref = img_ref.to('cpu')
        self.img_ref = img_ref.detach().numpy()[0]

        img_ref0 = net(camera_feed,torch.Tensor([[[[30]]]]).to('cuda:0)'))
        img_ref0 = img_ref0.to('cpu')
        self.img_ref0 = img_ref0.detach().numpy()[0]

        img_err =   (  -np.absolute(self.img_ref0 - self.img_ref) 
                       +np.absolute(self.img_ref0 - self.img)
                    )/255.0
        print(img_err)
        return img_err

def main():
    """
    The main method implements the game loop.
    """
    pygame.init()
    pygame.font.init()

    try:
        # Create the display
        display = pygame.display.set_mode(
                    (800, 600),
                    pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Car Following")

        # Create the carfollowing object
        envCar = CarFollowing()

        # This is the game loop
        clock = pygame.time.Clock()
        loop = True
        while loop:
            clock.tick_busy_loop(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    loop = False
            if envCar.surface is not None:
                display.blit(envCar.surface, (0, 0))
            
            if envCar.img is not None:
                img_err = envCar.getErrorSignal_BlockDiagram1(10)
                envCar.PControl(img_err)
            pygame.display.flip()

    finally:
        print('destroying actors')
        for actor in envCar.actor_list:
            actor.destroy()
        pygame.quit()
        np.savetxt('exp3_10.txt', envCar.distance)
        print('done.')


if __name__ == '__main__':
    main()

