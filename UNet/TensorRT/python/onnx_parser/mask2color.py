# -*- coding:utf-8 -*-

"""
    语义分割得到的 mask 结果，转换为带颜色的图像
"""

import numpy as np

classes_num = 32
idx2color_dict = {
    0: [0, 0, 0],  # Void
    1: [0, 0, 64],  # TrafficCone
    2: [0, 0, 192],  # Sidewalk
    3: [0, 64, 64],  # TrafficLight
    4: [0, 128, 64],  # Bridge
    5: [0, 128, 192],  # Bicyclist
    6: [64, 0, 64],  # Tunnel
    7: [64, 0, 128],  # Car
    8: [64, 0, 192],  # CartLuggagePram
    9: [64, 64, 0],  # Pedestrian
    10: [64, 64, 128],  # Fence
    11: [64, 128, 64],  # Animal
    12: [64, 128, 192],  # SUVPickupTruck
    13: [64, 192, 0],  # Wall
    14: [64, 192, 128],  # ParkingBlock
    15: [128, 0, 0],  # Building
    16: [128, 0, 192],  # LaneMkgsDriv
    17: [128, 64, 64],  # OtherMoving
    18: [128, 64, 128],  # Road
    19: [128, 128, 0],  # Tree
    20: [128, 128, 64],  # Misc_Text
    21: [128, 128, 128],  # Sky
    22: [128, 128, 192],  # RoadShoulder
    23: [192, 0, 64],  # LaneMkgsNonDriv
    24: [192, 0, 128],  # Archway
    25: [192, 0, 192],  # MotorcycleScooter
    26: [192, 64, 128],  # Train
    27: [192, 128, 64],  # Child
    28: [192, 128, 128],  # SignSymbol
    29: [192, 128, 192],  # Truck_Bus
    30: [192, 192, 0],  # VegetationMisc
    31: [192, 192, 128]  # Column_Pole
}


def to_color(mask):
    height, width = mask.shape
    color_map = np.zeros((height, width, 3), dtype=np.uint8)
    for cls in idx2color_dict:
        color_map[mask == cls] = idx2color_dict[cls]

    return color_map
