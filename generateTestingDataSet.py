#!/usr/bin/env python

# Author: Murad Abu-Khalaf, MIT CSAIL.

"""
    This generates a testing data set from CARLA to test generalization.

    The data set is in the form of camera views only.

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

class CarFollowing:
    def __init__(self):

        self.actor_list = []

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

            # Freeze traffic lights
            traffic_lights = [i for i in world.get_actors() if i.type_id == "traffic.traffic_light"]
            for i in traffic_lights:
                i.freeze(True)

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

            # Spawn the cars at the locations specified per run:
            #  - Town03: Follower (blue): starts @ x = -10.0, y = 40.0, Leader (red) starts @ x = -10, y = 50.0
            #  - Town03: Follower (blue): starts @ x = -60.0, y = 138.0, Leader (red) starts @ x = -50, y = 138.0
            #  - Town04: Follower (blue): starts @ x = 140, y = -170, Leader (blue) starts @ x = 150, y = -170
            #  - Town05: Follower (blue): starts @ x = 155.0, y = 85.0, Leader (red) starts @ x = 155.0, y = 75.0
            follower_transform = carla.Transform(carla.Location(x = -60.0, y = 138.50, z=2.5),
                carla.Rotation(pitch = 0, yaw = 0, roll = 0))
            leader_transform = carla.Transform(carla.Location(x = -50.0, y = 138.50, z=2.5),
                carla.Rotation(pitch = 0, yaw = 0, roll = 0))        
            self.follower = world.try_spawn_actor(follower_bp, follower_transform)
            self.leader = world.try_spawn_actor(leader_bp, leader_transform)

            # Storing created actors so we may destroy later.
            self.actor_list.append(self.follower)
            print('created %s' % self.follower.type_id)
            self.actor_list.append(self.leader)
            print('created %s' % self.leader.type_id)

            # Move the leader
            controlValue = carla.VehicleControl(throttle = 0.2,
                                                steer = 0,
                                                brake = 0,
                                                hand_brake = False,
                                                reverse = False,
                                                manual_gear_shift = False,
                                                gear = 1)
            self.leader.apply_control(controlValue)

            # Attach a camera to the follower
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('sensor_tick','0.5')
            camera_transform = carla.Transform(carla.Location(x=0.0, z=2.4))
            camera = world.spawn_actor(camera_bp, camera_transform, attach_to=self.follower)
            self.actor_list.append(camera)
            print('created %s' % camera.type_id)
            print(camera.attributes['sensor_tick'])
            camera.listen(self.sensorCallback)

            # Initialize distance
            self.distance = np.empty((0,2))

        finally:
            print("....")

      
    def sensorCallback(self,image):
        # Camera Callback
        loc = self.leader.get_location()
        d = np.sqrt((image.transform.location.x - loc.x)**2 + (image.transform.location.y - loc.y)**2)
        row = np.array([image.frame, d])
        self.distance = np.concatenate((self.distance, [row]))
        print(self.distance)    
        cc = carla.ColorConverter.Raw
        image.save_to_disk('CameraViewDistanceDataSet/TestingDataSet/%06d.png' % image.frame, cc)


def main():
    try:
        env = CarFollowing()
        time.sleep(180)
    finally:
        print('destroying actors')
        for actor in env.actor_list:
            actor.destroy()
        # No need for distances, only views needed
        #np.savetxt('CameraViewDistanceDataSet/TestingDataSet/distances.txt', env.distance)
        print('done.')


if __name__ == '__main__':
    main()
