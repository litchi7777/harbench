"""
Dataset Information Management

Defines metadata for each HAR dataset.
"""
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


# Dataset metadata
DATASETS = {
    "DSADS": {
        "sensor_list": ["Torso", "RightArm", "LeftArm", "RightLeg", "LeftLeg"],
        "modalities": ["ACC", "GYRO", "MAG"],
        "n_classes": 19,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 25,  # Hz
        "scale_factor": 9.8,  # m/s^2 -> G conversion (ACC only)
        "has_undefined_class": False,  # No label -1 (all samples have defined classes)
        "labels": {
            0: 'Sitting', 1: 'Standing', 2: 'Lying(Back)', 3: 'Lying(Right)',
            4: 'StairsUp', 5: 'StairsDown', 6: 'Standing(Elevator, still)',
            7: 'Moving(elevator)', 8: 'Walking(parking)',
            9: 'Walking(Treadmill, Flat)', 10: 'Walking(Treadmill, Slope)',
            11: 'Running(treadmill)', 12: 'Exercising(Stepper)',
            13: 'Exercising(Cross trainer)', 14: 'Cycling(Exercise bike, Vertical)',
            15: 'Cycling(Exercise bike, Horizontal)', 16: 'Rowing',
            17: 'Jumping', 18: 'PlayingBasketball'
        },
    },
    "HHAR": {
        "sensor_list": [
            "nexus4_1", "nexus4_2", "s3_1", "s3_2",
            "s3mini_1", "s3mini_2", "gear_1"
        ],  # samsungold, gear_2, lgwatch excluded due to missing data
        "modalities": ["ACC", "GYRO"],
        "n_classes": 6,
        "sampling_rate": 30,  # Hz (after resampling, unified with other datasets)
        "original_sampling_rate": None,  # Varies by device (50-200Hz)
        "scale_factor": 9.8,  # m/s^2 -> G conversion (ACC only)
        "has_undefined_class": True,  # gt='null' treated as -1
        "labels": {
            -1: "Undefined",
            0: "Bike",
            1: "Sit",
            2: "Stand",
            3: "Walk",
            4: "StairsUp",
            5: "StairsDown",
        },
    },
    "WISDM": {
        "sensor_list": ["Phone", "Watch"],
        "modalities": ["ACC", "GYRO"],
        "n_classes": 18,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 20,  # Hz
        "scale_factor": 9.8,  # m/s^2 -> G conversion (ACC only)
        "has_undefined_class": False,
        "labels": {
            0: "Walking", 1: "Jogging", 2: "Stairs",
            3: "Sitting", 4: "Standing", 5: "Typing",
            6: "BrushingTeeth", 7: "EatingSoup", 8: "EatingChips",
            9: "EatingPasta", 10: "Drinking", 11: "EatingSandwich",
            12: "Kicking", 13: "Catching", 14: "Dribbling",
            15: "Writing", 16: "Clapping", 17: "FoldingClothes"
        },
    },
    "REALWORLD": {
        "sensor_list": ["Chest", "Forearm", "Head", "Shin", "Thigh", "UpperArm", "Waist"],
        "modalities": ["ACC", "GYRO", "MAG"],
        "n_classes": 8,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 50,  # Hz
        "scale_factor": 9.8,  # m/s^2 -> G conversion (ACC only)
        "has_undefined_class": False,  # All samples have defined classes
        "labels": {
            0: 'ClimbingDown',
            1: 'ClimbingUp',
            2: 'Jumping',
            3: 'Lying',
            4: 'Running',
            5: 'Sitting',
            6: 'Standing',
            7: 'Walking'
        },
    },
    "PAMAP2": {
        "sensor_list": ["hand", "chest", "ankle"],
        "modalities": {
            "hand": ["ACC", "GYRO", "MAG"],
            "chest": ["ACC", "GYRO", "MAG"],
            "ankle": ["ACC", "GYRO", "MAG"]
        },
        "n_classes": 12,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 100,  # Hz
        "scale_factor": 9.8,  # m/s^2 -> G conversion (ACC only)
        "has_undefined_class": True,  # Label 0 (transient activities) converted to -1
        "labels": {
            -1: 'other',  # transient activities (original label 0)
            0: 'lying',   # original label 1
            1: 'sitting',  # original label 2
            2: 'standing',  # original label 3
            3: 'walking',  # original label 4
            4: 'running',  # original label 5
            5: 'cycling',  # original label 6
            6: 'Nordic walking',  # original label 7
            7: 'ascending stairs',  # original label 12
            8: 'descending stairs',  # original label 13
            9: 'vacuum cleaning',  # original label 16
            10: 'ironing',  # original label 17
            11: 'rope jumping'  # original label 24
        },
    },
    "MHEALTH": {
        "sensor_list": ["Chest", "LeftAnkle", "RightWrist"],
        "modalities": {
            "Chest": ["ACC", "ECG"],
            "LeftAnkle": ["ACC", "GYRO", "MAG"],
            "RightWrist": ["ACC", "GYRO", "MAG"]
        },
        "n_classes": 12,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 50,  # Hz
        "scale_factor": 9.8,  # m/s^2 -> G conversion (ACC only)
        "has_undefined_class": True,  # Label -1 exists
        "labels": {
            -1: 'Undefined',  # Undefined/no activity
            0: 'Standing', 1: 'Sitting', 2: 'LyingDown', 3: 'Walking',
            4: 'StairsUp', 5: 'WaistBendsForward', 6: 'FrontalElevationArms',
            7: 'KneesBending', 8: 'Cycling', 9: 'Jogging',
            10: 'Running', 11: 'JumpFrontBack'
        },
    },
    "OPENPACK": {
        "sensor_list": ["RightWrist", "LeftWrist", "RightUpperArm", "LeftUpperArm"],
        "modalities": {
            "RightWrist": ["ACC", "GYRO", "QUAT"],
            "LeftWrist": ["ACC", "GYRO", "QUAT"],
            "RightUpperArm": ["ACC", "GYRO", "QUAT"],
            "LeftUpperArm": ["ACC", "GYRO", "QUAT"]
        },  # Normalized body part names from atr01-04
        "n_classes": 9,  # 9 classes (0-8) + Undefined(-1)
        "sampling_rate": 30,  # Hz
        "original_sampling_rate": 30,  # Hz (no resampling needed)
        "has_undefined_class": True,  # Label -1 exists
        "labels": {
            -1: 'Undefined',  # Undefined/no operation/other (operation=0,10)
            0: 'Assemble', 1: 'Insert', 2: 'Put', 3: 'Walk',
            4: 'Pick', 5: 'Scan', 6: 'Press', 7: 'Open',
            8: 'Close'
        },
    },
    "FORTHTRACE": {
        "sensor_list": ["LeftWrist", "RightWrist", "Torso", "RightThigh", "LeftAnkle"],
        "modalities": ["ACC", "GYRO", "MAG"],
        "n_classes": 16,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 51.2,  # Hz
        "scale_factor": 9.8,  # m/s^2 -> G conversion (ACC only)
        "has_undefined_class": False,  # All samples have defined classes
        "labels": {
            0: 'Stand', 1: 'Sit', 2: 'Sit and Talk', 3: 'Walk',
            4: 'Walk and Talk', 5: 'Climb Stairs', 6: 'Climb Stairs and Talk',
            7: 'Stand -> Sit', 8: 'Sit -> Stand', 9: 'Stand -> Sit and Talk',
            10: 'Sit and Talk -> Stand', 11: 'Stand -> Walk', 12: 'Walk -> Stand',
            13: 'Stand -> Climb Stairs', 14: 'Climb Stairs -> Walk',
            15: 'Climb Stairs and Talk -> Walk and Talk'
        },
    },
    "HAR70PLUS": {
        "sensor_list": ["LowerBack", "RightThigh"],
        "modalities": ["ACC"],  # Accelerometer only (3-axis)
        "n_classes": 7,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 50,  # Hz
        "scale_factor": None,  # Already in G units, no conversion needed
        "has_undefined_class": False,  # All samples have defined classes
        "labels": {
            0: 'Walking',       # label=1 -> 0
            1: 'Shuffling',     # label=3 -> 1
            2: 'Stairs Up',     # label=4 -> 2
            3: 'Stairs Down',   # label=5 -> 3
            4: 'Standing',      # label=6 -> 4
            5: 'Sitting',       # label=7 -> 5
            6: 'Lying'          # label=8 -> 6
        },
    },
    "HARTH": {
        "sensor_list": ["LowerBack", "RightThigh"],
        "modalities": ["ACC"],  # Accelerometer only (3-axis)
        "n_classes": 12,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 50,  # Hz
        "scale_factor": None,  # Already in G units, no conversion needed
        "has_undefined_class": False,  # All samples have defined classes
        "labels": {
            0: 'Walking',                   # label=1 -> 0
            1: 'Running',                   # label=2 -> 1 (not in sample, but in spec)
            2: 'Shuffling',                 # label=3 -> 2
            3: 'Stairs Up',                 # label=4 -> 3
            4: 'Stairs Down',               # label=5 -> 4
            5: 'Standing',                  # label=6 -> 5
            6: 'Sitting',                   # label=7 -> 6
            7: 'Lying',                     # label=8 -> 7
            8: 'Cycling Seated',            # label=13 -> 8
            9: 'Cycling Standing',          # label=14 -> 9
            10: 'Cycling Seated Inactive',  # label=130 -> 10
            11: 'Cycling Standing Inactive' # label=140 -> 11 (not in sample, but in spec)
        },
    },
    "LARA": {
        "sensor_list": ["LeftArm", "LeftLeg", "Neck", "RightArm", "RightLeg"],
        "modalities": ["ACC", "GYRO"],  # Accelerometer (3-axis) + Gyroscope (3-axis)
        "n_classes": 8,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 100,  # Hz
        "scale_factor": None,  # Already in G units, no conversion needed
        "has_undefined_class": False,  # All samples have defined classes
        "labels": {
            0: 'Stationary',
            1: 'GaitCycle',
            2: 'Step',
            3: 'Upwards',
            4: 'Centred',
            5: 'Downwards',
            6: 'TorsoRotation',
            7: 'OtherMotion'
        },
    },
    "REALDISP": {
        "sensor_list": ["LeftCalf", "LeftThigh", "RightCalf", "RightThigh", "Back",
                       "LeftLowerArm", "LeftUpperArm", "RightLowerArm", "RightUpperArm"],
        "modalities": ["ACC", "GYRO", "MAG", "QUAT"],  # Common for all sensors
        "n_classes": 33,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 50,  # Hz (estimated)
        "scale_factor": 9.8,  # m/s^2 -> G conversion (ACC only)
        "has_undefined_class": False,  # All samples have defined classes
        "labels": {
            0: 'Walking', 1: 'Jogging', 2: 'Running', 3: 'Jump up',
            4: 'Jump front & back', 5: 'Jump sideways', 6: 'Jump leg/arms open/closed',
            7: 'Jump rope', 8: 'Trunk twist (arms outstretched)', 9: 'Trunk twist (elbows bent)',
            10: 'Waist bends forward', 11: 'Waist rotation',
            12: 'Waist bends (reach foot with opposite hand)', 13: 'Reach heels backwards',
            14: 'Lateral bend (left+right)', 15: 'Lateral bend with arm up (left+right)',
            16: 'Repetitive forward stretching', 17: 'Upper trunk and lower body opposite twist',
            18: 'Lateral elevation of arms', 19: 'Frontal elevation of arms',
            20: 'Frontal hand claps', 21: 'Frontal crossing of arms',
            22: 'Shoulders high-amplitude rotation', 23: 'Shoulders low-amplitude rotation',
            24: 'Arms inner rotation', 25: 'Knees (alternating) to the breast',
            26: 'Heels (alternating) to the backside', 27: 'Knees bending (crouching)',
            28: 'Knees (alternating) bending forward', 29: 'Rotation on the knees',
            30: 'Rowing', 31: 'Elliptical bike', 32: 'Cycling'
        },
    },
    "MEX": {
        "sensor_list": ["Wrist", "Thigh"],
        "modalities": ["ACC"],  # Accelerometer only (3-axis)
        "n_classes": 7,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 100,  # Hz
        "scale_factor": None,  # Already in G units (±8g), no conversion needed
        "has_undefined_class": False,  # All samples have defined classes
        "labels": {
            0: 'Knee-rolling',
            1: 'Bridging',
            2: 'Pelvic tilt',
            3: 'The Clam',
            4: 'Repeated Extension in Lying',
            5: 'Prone punches',
            6: 'Superman'
        },
    },
    "OPPORTUNITY": {
        "sensor_list": ["BACK", "RUA", "RLA", "LUA", "LLA", "L_SHOE", "R_SHOE"],
        "modalities": {
            "BACK": ["ACC", "GYRO", "MAG"],
            "RUA": ["ACC", "GYRO", "MAG"],
            "RLA": ["ACC", "GYRO", "MAG"],
            "LUA": ["ACC", "GYRO", "MAG"],
            "LLA": ["ACC", "GYRO", "MAG"],
            "L_SHOE": ["ACC", "GYRO", "MAG"],
            "R_SHOE": ["ACC", "GYRO", "MAG"],
        },
        "n_classes": 17,  # Number of valid mid-level gesture classes
        "sampling_rate": 30,  # Hz (already 30Hz, no resampling needed)
        "original_sampling_rate": 30,  # Hz
        "scale_factor": 9.8,  # m/s² -> G conversion (ACC only)
        "has_undefined_class": True,  # Label -1 (Null class) exists
        "labels": {
            -1: 'Null',  # Undefined/no operation
            0: 'Open Door 1',
            1: 'Open Door 2',
            2: 'Close Door 1',
            3: 'Close Door 2',
            4: 'Open Fridge',
            5: 'Close Fridge',
            6: 'Open Dishwasher',
            7: 'Close Dishwasher',
            8: 'Open Drawer 1',
            9: 'Close Drawer 1',
            10: 'Open Drawer 2',
            11: 'Close Drawer 2',
            12: 'Open Drawer 3',
            13: 'Close Drawer 3',
            14: 'Clean Table',
            15: 'Drink from Cup',
            16: 'Toggle Switch'
        },
    },
    "USCHAD": {
        "sensor_list": ["Phone"],
        "modalities": ["ACC", "GYRO"],  # Accelerometer (3-axis) + Gyroscope (3-axis)
        "n_classes": 12,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 100,  # Hz
        "scale_factor": None,  # Already in G units, no conversion needed
        "has_undefined_class": False,  # All samples have defined classes
        "labels": {
            0: 'Walking Forward',
            1: 'Walking Left',
            2: 'Walking Right',
            3: 'Walking Upstairs',
            4: 'Walking Downstairs',
            5: 'Running Forward',
            6: 'Jumping Up',
            7: 'Sitting',
            8: 'Standing',
            9: 'Sleeping',
            10: 'Elevator Up',
            11: 'Elevator Down'
        },
    },
    "SELFBACK": {
        "sensor_list": ["Wrist", "Thigh"],
        "modalities": ["ACC"],  # Accelerometer only (3-axis)
        "n_classes": 9,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 100,  # Hz
        "scale_factor": None,  # Already in G units (±8g), no conversion needed
        "has_undefined_class": False,  # All samples have defined classes
        "labels": {
            0: 'Walking Downstairs',
            1: 'Walking Upstairs',
            2: 'Walking Slow',
            3: 'Walking Moderate',
            4: 'Walking Fast',
            5: 'Jogging',
            6: 'Sitting',
            7: 'Standing',
            8: 'Lying'
        },
    },
    "PAAL": {
        "sensor_list": ["Phone"],
        "modalities": ["ACC"],  # Accelerometer only (3-axis)
        "n_classes": 24,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 32,  # Hz
        "scale_factor": 0.015,  # Integer to G conversion (8-bit resolution: ±2g)
        "has_undefined_class": False,  # All samples have defined classes
        "labels": {
            0: 'Blow Nose',
            1: 'Brush Hair',
            2: 'Brush Teeth',
            3: 'Drink Water',
            4: 'Dusting',
            5: 'Eat Meal',
            6: 'Ironing',
            7: 'Open Bottle',
            8: 'Open Box',
            9: 'Phone Call',
            10: 'Put On Jacket',
            11: 'Put On Shoe',
            12: 'Put On Glasses',
            13: 'Salute',
            14: 'Sit Down',
            15: 'Sneeze/Cough',
            16: 'Stand Up',
            17: 'Take Off Jacket',
            18: 'Take Off Shoe',
            19: 'Take Off Glasses',
            20: 'Type On Keyboard',
            21: 'Washing Dishes',
            22: 'Washing Hands',
            23: 'Writing'
        },
    },
    "TMD": {
        "sensor_list": ["Smartphone"],  # Single smartphone sensor
        "modalities": ["ACC", "GYRO"],  # Accelerometer and gyroscope
        "n_classes": 5,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": None,  # Variable sampling rate (event-driven)
        "scale_factor": 9.8,  # m/s^2 -> G conversion (Android sensors are in m/s^2)
        "has_undefined_class": False,  # All samples have defined classes
        "labels": {
            0: 'Walking',
            1: 'Car',
            2: 'Still',
            3: 'Train',
            4: 'Bus'
        },
    },
    "WARD": {
        "sensor_list": ["LeftArm", "RightArm", "Waist", "LeftAnkle", "RightAnkle"],
        "modalities": ["ACC", "GYRO"],  # Accelerometer (3-axis) + Gyroscope (2-axis)
        "n_classes": 13,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 20,  # Hz
        "scale_factor": 1024.0,  # 12-bit digital values (±2g) -> G conversion
        "has_undefined_class": False,  # All samples have defined classes
        "labels": {
            0: 'Stand',
            1: 'Sit',
            2: 'Lie',
            3: 'Walk Forward',
            4: 'Walk Left-Circle',
            5: 'Walk Right-Circle',
            6: 'Turn Left',
            7: 'Turn Right',
            8: 'Go Upstairs',
            9: 'Go Downstairs',
            10: 'Jog',
            11: 'Jump',
            12: 'Push Wheelchair'
        },
    },
    "ADLRD": {
        "sensor_list": ["RightWrist"],  # Single 3-axis accelerometer on right wrist
        "modalities": ["ACC"],  # Accelerometer only (3-axis)
        "n_classes": 14,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 32,  # Hz
        "scale_factor": 63.0 / 3.0,  # 6-bit coded values (0-63 = -1.5g to +1.5g) -> G conversion
        "has_undefined_class": False,  # All samples have defined classes
        "labels": {
            0: 'Brush Teeth',
            1: 'Climb Stairs',
            2: 'Comb Hair',
            3: 'Descend Stairs',
            4: 'Drink Glass',
            5: 'Eat Meat',
            6: 'Eat Soup',
            7: 'Getup Bed',
            8: 'Liedown Bed',
            9: 'Pour Water',
            10: 'Sitdown Chair',
            11: 'Standup Chair',
            12: 'Use Telephone',
            13: 'Walk'
        },
    },
    "CAPTURE24": {
        "sensor_list": ["Wrist"],  # Wrist-worn accelerometer
        "modalities": ["ACC"],  # 3-axis acceleration only
        "n_classes": 10,  # WillettsSpecific2018 schema
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 100,  # Hz (Axivity AX3)
        "scale_factor": None,  # Already in G units
        "has_undefined_class": True,  # Some windows have NA annotations
        "labels": {
            0: 'bicycling',
            1: 'household-chores',
            2: 'manual-work',
            3: 'mixed-activity',
            4: 'sitting',
            5: 'sleep',
            6: 'sports',
            7: 'standing',
            8: 'vehicle',
            9: 'walking'
        },
        "notes": "Large-scale daily living dataset (151 participants, ~4000 hours). Using WillettsSpecific2018 schema."
    },
    "IMSB": {
        "sensor_list": ["Wrist", "Neck"],  # Thigh excluded due to high missing values
        "modalities": ["ACC"],  # 3-axis acceleration only
        "n_classes": 6,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 20,  # Hz (estimated: 1000samples/50s)
        "scale_factor": 9.8,  # m/s^2 -> G conversion (ACC only)
        "has_undefined_class": False,  # All samples have valid classes
        "labels": {
            0: 'Badminton',
            1: 'Basketball',
            2: 'Cycling',
            3: 'Football',
            4: 'Skipping',
            5: 'TableTennis'
        },
        "notes": "IM-SportingBehaviors dataset. 20 subjects, 6 sports. Thigh sensor excluded due to 30% missing values."
    },
    "MOTIONSENSE": {
        "sensor_list": ["Pocket"],  # iPhone 6s in front pocket
        "modalities": {
            "Pocket": ["ATT", "GRA", "ROT", "ACC"]  # attitude, gravity, rotationRate, userAcceleration
        },
        "n_classes": 6,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 50,  # Hz
        "scale_factor": None,  # Already in G units (userAcceleration)
        "has_undefined_class": False,  # All samples have valid classes
        "labels": {
            0: 'Downstairs',
            1: 'Upstairs',
            2: 'Walking',
            3: 'Jogging',
            4: 'Standing',
            5: 'Sitting'
        },
        "notes": "MotionSense dataset. 24 subjects, iPhone 6s DeviceMotion data (attitude+gravity+rotation+userAcceleration)."
    },
    "IMWSHA": {
        "sensor_list": ["Wrist", "Chest", "Thigh"],  # MPU-9250 IMU sensors
        "modalities": ["ACC", "GYRO", "MAG"],  # Accelerometer + Gyroscope + Magnetometer
        "n_classes": 11,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 50,  # Hz (estimated from avg norm 9.8 m/s²)
        "scale_factor": 9.8,  # m/s^2 -> G conversion (ACC only)
        "has_undefined_class": False,  # All samples have valid classes
        "labels": {
            0: 'Using Computer',
            1: 'Phone Conversation',
            2: 'Vacuum Cleaning',
            3: 'Reading Book',
            4: 'Watching TV',
            5: 'Ironing',
            6: 'Walking',
            7: 'Exercise',
            8: 'Cooking',
            9: 'Drinking',
            10: 'Brushing Hair'
        },
        "notes": "IM-Wearable Smart Home Activities. 10 subjects, 11 smart home activities, 3 IMU sensors (9-axis each)."
    },
    "SBRHAPT": {
        "sensor_list": ["Waist"],  # Samsung Galaxy S II on waist
        "modalities": ["ACC", "GYRO"],  # Accelerometer + Gyroscope
        "n_classes": 12,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 50,  # Hz
        "scale_factor": None,  # Already in G units (-1 to 1 range)
        "has_undefined_class": False,  # All samples have valid classes
        "labels": {
            0: 'Walking',
            1: 'Walking Upstairs',
            2: 'Walking Downstairs',
            3: 'Sitting',
            4: 'Standing',
            5: 'Laying',
            6: 'Stand to Sit',
            7: 'Sit to Stand',
            8: 'Sit to Lie',
            9: 'Lie to Sit',
            10: 'Stand to Lie',
            11: 'Lie to Stand'
        },
        "notes": "Smartphone-Based Recognition of Human Activities and Postural Transitions. 30 subjects, 6 basic activities + 6 postural transitions."
    },
    "CHAD": {
        "sensor_list": ["Pocket"],  # Smartphone (pocket only, Wrist has missing data)
        "modalities": {
            "Pocket": ["ACC", "LINACC", "GYRO", "MAG"],
        },
        "n_classes": 13,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 50,  # Hz (estimated from paper)
        "scale_factor": 9.8,  # m/s² -> G conversion
        "has_undefined_class": False,  # All samples have defined classes
        "labels": {
            0: 'Walk',
            1: 'Stand',
            2: 'Jog',
            3: 'Sit',
            4: 'Bike',
            5: 'Upstairs',
            6: 'Downstairs',
            7: 'Type',
            8: 'Write',
            9: 'Coffee',
            10: 'Talk',
            11: 'Smoke',
            12: 'Eat'
        },
    },
    "UCAEHAR": {
        "sensor_list": ["SmartGlasses"],  # Sensors embedded in smart glasses
        "modalities": ["ACC", "GYRO", "BAR"],  # Accelerometer + Gyroscope + Barometer
        "n_classes": 8,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 25,  # Hz (estimated from timestamps)
        "scale_factor": 9.8,  # m/s² -> G conversion (ACC only)
        "has_undefined_class": True,  # Posture transitions (4 classes) excluded as -1
        "labels": {
            -1: 'PostureTransition',  # STAND_TO_SIT, SIT_TO_STAND, SIT_TO_LIE, LIE_TO_SIT
            0: 'Walking',
            1: 'Running',
            2: 'Standing',
            3: 'Sitting',
            4: 'Lying',
            5: 'Drinking',
            6: 'WalkingUpstairs',
            7: 'WalkingDownstairs'
        },
        "notes": "UCA-EHAR dataset from Université Côte d'Azur. 20 subjects, 8 activities. Smart glasses with accelerometer, gyroscope, and barometer. Posture transitions excluded."
    },
    "DOG": {
        "sensor_list": ["Ankle", "Thigh", "Trunk"],  # 3 sensor locations
        "modalities": ["ACC"],  # Accelerometer only (3-axis)
        "n_classes": 2,  # Freezing of Gait detection: No Freeze, Freeze
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 64,  # Hz
        "scale_factor": 1000.0,  # mg -> G conversion (milliG units)
        "has_undefined_class": True,  # Label 0 (non-experiment) converted to -1
        "labels": {
            -1: 'NonExperiment',  # Non-experiment (setup/debriefing)
            0: 'NoFreeze',        # During experiment, no freeze (Stand/Walk/Turn)
            1: 'Freeze'           # Freezing of Gait event
        },
        "notes": "Daphnet Freezing of Gait dataset. Parkinson's disease patients. 10 subjects, 3 sensors. Used for Freezing of Gait detection."
    },
    "KDDI_KITCHEN_LEFT": {
        "sensor_list": ["Wrist"],  # Left wrist smartwatch
        "modalities": ["ACC"],  # Accelerometer only (3-axis)
        "n_classes": 24,  # 24 cooking activities
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 64,  # Hz
        "scale_factor": 9.8,  # m/s² -> G conversion
        "has_undefined_class": True,  # class 0 converted to -1
        "labels": {
            -1: "Undefined",  # Original class 0: other/no activity
            0: "Activity_1",
            1: "Activity_2",
            2: "Activity_4",
            3: "Activity_5",
            4: "Activity_10",
            5: "Activity_19",
            6: "Activity_20",
            7: "Activity_29",
            8: "Activity_38",
            9: "Activity_39",
            10: "Activity_42",
            11: "Activity_43",
            12: "Activity_45",
            13: "Activity_51",
            14: "Activity_53",
            15: "Activity_55",
            16: "Activity_58",
            17: "Activity_63",
            18: "Activity_64",
            19: "Activity_65",
            20: "Activity_67",
            21: "Activity_68",
            22: "Activity_69",
            23: "Activity_72",
        },
        "notes": "KDDI Kitchen Smartwatch Dataset (Left Arm). 10 subjects, cooking activities. MUM 2017."
    },
    "KDDI_KITCHEN_RIGHT": {
        "sensor_list": ["Wrist"],  # Right wrist smartwatch
        "modalities": ["ACC"],  # Accelerometer only (3-axis)
        "n_classes": 24,  # 24 cooking activities
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 64,  # Hz
        "scale_factor": 9.8,  # m/s² -> G conversion
        "has_undefined_class": True,  # class 0 converted to -1
        "labels": {
            -1: "Undefined",  # Original class 0: other/no activity
            0: "Activity_1",
            1: "Activity_2",
            2: "Activity_3",
            3: "Activity_4",
            4: "Activity_5",
            5: "Activity_7",
            6: "Activity_8",
            7: "Activity_19",
            8: "Activity_28",
            9: "Activity_29",
            10: "Activity_37",
            11: "Activity_38",
            12: "Activity_39",
            13: "Activity_40",
            14: "Activity_50",
            15: "Activity_53",
            16: "Activity_59",
            17: "Activity_60",
            18: "Activity_61",
            19: "Activity_62",
            20: "Activity_63",
            21: "Activity_64",
            22: "Activity_65",
            23: "Activity_66",
        },
        "notes": "KDDI Kitchen Smartwatch Dataset (Right Arm). 10 subjects, cooking activities. MUM 2017."
    },
    "VTT_CONIOT": {
        "sensor_list": ["Hip", "Back", "UpperArm"],  # trousers(hip), back, hand(upper arm)
        "modalities": ["ACC", "GYRO", "MAG"],  # Accelerometer + Gyroscope + Magnetometer
        "n_classes": 16,
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 100,  # Hz
        "scale_factor": 0.5,  # half-G -> G conversion (ACC only)
        "has_undefined_class": False,  # All samples have defined classes
        "labels": {
            0: 'RollPainting',          # Roll Painting: paint-roll on wall
            1: 'SprayingPaint',         # Spraying Paint: spray movements
            2: 'LevellingPaint',        # Levelling paint: spreading screed/paint
            3: 'VacuumCleaning',        # Vacuum Cleaning
            4: 'PickingObjects',        # Picking objects from floor
            5: 'ClimbingStairs',        # Climbing stairs (up/down 3 steps)
            6: 'JumpingDown',           # Jumping down 3 steps
            7: 'LayingBack',            # Working with hands up while laying back
            8: 'HandsupHigh',           # Hands up above head (working on tubes)
            9: 'HandsupLow',            # Hands up at head/shoulder level
            10: 'CrouchFloor',          # Working on floor while crouching
            11: 'KneelFloor',           # Working on floor while kneeling
            12: 'WalkStraight',         # Walking straight 20m
            13: 'WalkWinding',          # Walking winding around cones
            14: 'PushingCart',          # Pushing cart 20m
            15: 'StairsUpDown'          # Climbing stairs for 30s
        },
        "notes": "VTT-ConIot Construction Worker HAR. 13 subjects, 16 construction activities. 3 IMUs (hip, back, upper arm). CC BY 4.0."
    },
    "EXOSKELETONS": {
        "sensor_list": ["Chest", "RightLeg", "LeftLeg", "RightWrist", "LeftWrist"],
        "modalities": ["ACC", "GYRO"],  # Accelerometer + Gyroscope (each 3-axis)
        "n_classes": 4,  # Intention task only
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 50,  # Hz (estimated)
        "scale_factor": 1.0 / 9.80665,  # m/s² -> G conversion (ACC only)
        "has_undefined_class": False,  # All samples have valid classes
        "labels": {
            0: 'Idle',
            1: 'Walking',
            2: 'Lifting',
            3: 'Lowering'
        },
        "notes": "IMU-based HAR for Low-Back Exoskeletons. 12 subjects (6M+6F), 5 IMUs. Intention task only (payload task excluded). Label mapping inferred from paper description. Zenodo: 7182799."
    },
    "NHANES": {
        "sensor_list": ["Waist"],  # Single waist-worn accelerometer
        "modalities": ["ACC"],  # Accelerometer only (3-axis)
        "n_classes": 0,  # Unlabeled dataset
        "sampling_rate": 30,  # Hz (after resampling)
        "original_sampling_rate": 80,  # Hz
        "scale_factor": None,  # Already in G units
        "has_undefined_class": True,  # All samples are unlabeled
        "labels": {},  # No activity labels
        "notes": "NHANES Physical Activity Monitor dataset. ~13,000 participants, 7-day continuous wear. Unlabeled accelerometer data for population-level physical activity research."
    },
}


def get_available_sensors(data_root: str) -> List[str]:
    """
    Get available sensor locations in the dataset

    Args:
        data_root: Data root path

    Returns:
        List of available sensor locations
    """
    data_path = Path(data_root)

    if not data_path.exists():
        raise ValueError(f"Data root not found: {data_root}")

    # Search subdirectories
    sensors = []
    for item in data_path.iterdir():
        if item.is_dir():
            # Check if X.npy, Y.npy exist
            required_files = ['X.npy', 'Y.npy']
            if all((item / f).exists() for f in required_files):
                sensors.append(item.name)

    return sorted(sensors)


def get_dataset_info(dataset_name: str, data_root: str = None) -> Dict[str, any]:
    """
    Get basic dataset information

    Args:
        dataset_name: Dataset name (e.g., "DSADS")
        data_root: Data root path (for getting info from actual data)

    Returns:
        Dictionary of dataset information
    """
    # Get info from metadata
    if dataset_name in DATASETS:
        meta = DATASETS[dataset_name]
        info = {
            'dataset_name': dataset_name,
            'sensor_list': meta['sensor_list'],
            'n_classes': meta['n_classes'],
            'labels': meta['labels'],
        }

        # If data_root is specified, get additional info from actual data
        if data_root:
            available_sensors = get_available_sensors(data_root)
            info['available_sensors'] = available_sensors

            # Get data size from first sensor
            if available_sensors:
                sensor_path = Path(data_root) / available_sensors[0]
                Y = np.load(sensor_path / "Y.npy")
                X = np.load(sensor_path / "X.npy")

                # Get user count from directory structure
                user_dirs = [d.name for d in Path(data_root).parent.iterdir()
                            if d.is_dir() and d.name.startswith('USER')]
                info['num_users'] = len(user_dirs)
                info['user_ids'] = sorted(user_dirs)
                info['num_samples'] = len(Y)
                info['channels_per_sensor'] = X.shape[1] if len(X.shape) > 1 else 1
                info['sequence_length'] = X.shape[2] if len(X.shape) > 2 else X.shape[1]

        return info
    else:
        # If dataset not defined in metadata, detect from actual data
        if not data_root:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in metadata. "
                f"Provide 'data_root' to auto-detect."
            )

        available_sensors = get_available_sensors(data_root)
        if not available_sensors:
            raise ValueError(f"No valid sensor data found in {data_root}")

        # Get info from first sensor
        sensor_path = Path(data_root) / available_sensors[0]
        Y = np.load(sensor_path / "Y.npy")
        X = np.load(sensor_path / "X.npy")

        # Get user count from directory structure
        user_dirs = [d.name for d in Path(data_root).parent.iterdir()
                    if d.is_dir() and d.name.startswith('USER')]

        return {
            'dataset_name': dataset_name,
            'available_sensors': available_sensors,
            'sensor_list': available_sensors,
            'num_users': len(user_dirs),
            'user_ids': sorted(user_dirs),
            'num_samples': len(Y),
            'n_classes': len(np.unique(Y)),
            'channels_per_sensor': X.shape[1] if len(X.shape) > 1 else 1,
            'sequence_length': X.shape[2] if len(X.shape) > 2 else X.shape[1],
        }


def select_sensors(
    dataset_name: str,
    data_root: str,
    mode: str,
    specific_sensors: Optional[List[str]] = None
) -> List[str]:
    """
    Select sensor locations to use

    Args:
        dataset_name: Dataset name
        data_root: Data root path
        mode: "single_device" or "multi_device"
        specific_sensors: List of specific sensor locations (optional)

    Returns:
        List of sensor locations to use
    """
    available_sensors = get_available_sensors(data_root)

    if specific_sensors:
        # Check if specified sensors are available
        for s in specific_sensors:
            if s not in available_sensors:
                raise ValueError(
                    f"Sensor '{s}' not available. "
                    f"Available sensors: {available_sensors}"
                )
        return specific_sensors

    if mode == "single_device":
        # Use first sensor if metadata exists
        if dataset_name in DATASETS:
            first_sensor = DATASETS[dataset_name]['sensor_list'][0]
            if first_sensor in available_sensors:
                return [first_sensor]
        # Otherwise use first available sensor
        return [available_sensors[0]]

    elif mode == "multi_device":
        # Use defined sensor list if metadata exists
        if dataset_name in DATASETS:
            meta_sensors = DATASETS[dataset_name]['sensor_list']
            # Filter to only available sensors
            return [s for s in meta_sensors if s in available_sensors]
        # Otherwise use all sensors
        return available_sensors

    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'single_device' or 'multi_device'")
