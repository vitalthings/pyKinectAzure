# %%
import numpy as np
import matplotlib.pyplot as plt
from rotation import closest_rotation_matrix
import json
import glob
import argparse
import os
from scipy.ndimage import gaussian_filter1d

INTERPOLATE_MISSING = False

parser = argparse.ArgumentParser(description="Plot positions from multi-cam tracking and generate ground truth.")
parser.add_argument('-c', '--calibration-folder', default='data/calibration/cal_0005/', help='Path to calibration folder (e.g. data/calibration/cal_0005)')
parser.add_argument('-t', '--tracking-folder', default='data/tracking/movement_still/', help='Path to tracking folder (e.g. data/tracking/track_0004)')
parser.add_argument('-o', '--ground-truth-output-folder', default='.', help='Directory to store ground_truth.npz')
parser.add_argument('-j', '--joint-id', type=int, default=26, help='Joint ID to extract (default: 26)')
parser.add_argument('--skip-plot', action='store_true', help='Skip plotting (only compute and save ground truth)')
args = parser.parse_args()

calibration_folder = args.calibration_folder
tracking_folder = args.tracking_folder
ground_truth_output_folder = args.ground_truth_output_folder

# emblobot in room coordinates
c0_room = np.array([[2.05, 2.38, 1.22]]).T * 1000

# %%
def reject_outliers(data, m=2.):
    d = np.abs(data - np.mean(data, axis=0))
    d_norm = np.linalg.norm(d, axis=1)
    outlier_idxs = np.where(d_norm > m * np.mean(d_norm))
    return np.delete(data, outlier_idxs, axis=0)

# Load calibration data for all cameras
calibration_files = glob.glob(f"{calibration_folder}/calibration_data_cam*.json")

camera_calibrations = {}
for file in calibration_files:
    with open(file, 'r') as f:
        calibration_data = json.load(f)
    cam_id = file.split('_cam')[-1].split('.json')[0]  # Extract camera ID from filename
    camera_calibrations[cam_id] = {
        "c0": np.array([calibration_data[i]["c0"] for i in range(len(calibration_data))]),
        "x_vec": np.array([calibration_data[i]["x_vec"] for i in range(len(calibration_data))]),
        "y_vec": np.array([calibration_data[i]["y_vec"] for i in range(len(calibration_data))]),
        "z_vec": np.array([calibration_data[i]["z_vec"] for i in range(len(calibration_data))]),
    }

#%%
# Reject outliers for calibration data
for cam_id, calib in camera_calibrations.items():
    calib["c0"] = reject_outliers(calib["c0"])
    calib["x_vec"] = reject_outliers(calib["x_vec"])
    calib["y_vec"] = reject_outliers(calib["y_vec"])
    calib["z_vec"] = reject_outliers(calib["z_vec"])

# %%
# Load tracking data for all cameras
tracking_files = glob.glob(f"{tracking_folder}/track_data_cam*.json")

camera_data = {}
camera_ids = [file.split('_cam')[-1].split('.json')[0] for file in tracking_files]  # Extract camera IDs

for file in tracking_files:
    with open(file, 'r') as f:
        cam_id = file.split('_cam')[-1].split('.json')[0]
        camera_data[cam_id] = json.load(f)

# Helper functions
def get_joint_positions(json_data, joint_id):
    return np.array(
        [
            (
                json_data[i]["joints"][joint_id]["position"]["v"]
                if json_data[i]["num_bodies"] > 0
                else [0, 0, 0]
            )
            for i in range(len(json_data))
        ]
    )

def get_joint_confidences(json_data, joint_id):
    return np.array(
        [
            (
                json_data[i]["joints"][joint_id]["confidence_level"]
                if json_data[i]["num_bodies"] > 0
                else 0
            )
            for i in range(len(json_data))
        ]
    )

def get_timestamps(json_data):
    return np.array(
        [
            json_data[i]["device_timestamp_us"]
            for i in range(len(json_data))
        ]
    )

def get_utc_timestamps(json_data):
    return np.array(
        [
            json_data[i]["utc_timestamp_us"]
            for i in range(len(json_data))
        ]
    )

# Process joint data for all cameras
# joint_id = 26  # Example: head_joint_id
joint_id = args.joint_id  # Selected via CLI

camera_positions = {}
camera_confidences = {}
camera_timestamps = {}
camera_tracks = {}

common_utc_timestamps = get_utc_timestamps(camera_data[camera_ids[0]])

for cam_id, json_data in camera_data.items():
    camera_positions[cam_id] = get_joint_positions(json_data, joint_id)
    camera_confidences[cam_id] = get_joint_confidences(json_data, joint_id)
    camera_timestamps[cam_id] = get_timestamps(json_data)
    camera_tracks[cam_id] = np.array([json_data[i]["num_bodies"] > 0 for i in range(len(json_data))])

# %%
# Compute mean calibration vectors and rotation matrices for all cameras
camera_means = {}
camera_rotations = {}

for cam_id, calib in camera_calibrations.items():
    c_mean = np.mean(calib["c0"], axis=0)
    x_vec_mean = np.mean(calib["x_vec"], axis=0)
    y_vec_mean = np.mean(calib["y_vec"], axis=0)
    z_vec_mean = np.mean(calib["z_vec"], axis=0)
    R = np.array([x_vec_mean, y_vec_mean, z_vec_mean]).T
    R_SVD = closest_rotation_matrix(R)
    camera_means[cam_id] = c_mean
    camera_rotations[cam_id] = R_SVD

# Transform positions to room coordinates
camera_room_positions = {}

for cam_id, positions in camera_positions.items():
    R_SVD = camera_rotations[cam_id]
    c_mean = camera_means[cam_id]
    room_positions = R_SVD.T @ (positions - c_mean).T
    room_positions = room_positions + c0_room
    room_positions /= 1000  # Convert to meters
    camera_room_positions[cam_id] = room_positions.T

# %%
# Handle missing tracks by replacing with closest timestamps from other cameras
def get_closest_timestamp_index(timestamps, target_timestamp):
    closest_index = (np.abs(timestamps - target_timestamp)).argmin()
    return closest_index

reference_frame = 10
cam_frame_offsets = {'0': reference_frame}
min_length = len(camera_timestamps['0']) - reference_frame
for cam_id, timestamps in camera_timestamps.items():
    if cam_id == '0':
        continue
    cam_frame_offsets[cam_id] = get_closest_timestamp_index(timestamps, camera_timestamps['0'][reference_frame])
    min_length = min(min_length, len(timestamps) - cam_frame_offsets[cam_id])

print("min length:", min_length)
print("Camera frame offsets:")
for cam_id, offset in cam_frame_offsets.items():
    print(f"Camera {cam_id}: {offset}")
    print(f"Camera {cam_id} timestamps: {camera_timestamps[cam_id][offset]}")


#%%
# Align and trim data based on calculated offsets and minimum length
common_utc_timestamps = common_utc_timestamps[reference_frame:reference_frame + min_length]
timestamps = (common_utc_timestamps - common_utc_timestamps[0]) / 1e6

for cam_id in camera_ids:
    offset = cam_frame_offsets[cam_id]
    camera_room_positions[cam_id] = camera_room_positions[cam_id][offset:offset + min_length]
    camera_tracks[cam_id] = camera_tracks[cam_id][offset:offset + min_length]
    camera_confidences[cam_id] = camera_confidences[cam_id][offset:offset + min_length]

#%%

# make a superposition of all cameras, which is the average of all cameras that have a track
camera_superposition = np.zeros((min_length, 3))
camera_superposition_track = np.zeros((min_length, 1))
for i in range(min_length):
    count = 0
    for cam_id in camera_ids:
        if camera_tracks[cam_id][i]:
            camera_superposition[i] += camera_room_positions[cam_id][i]
            camera_superposition_track[i] += 1
            count += 1
    if count > 0:
        camera_superposition[i] /= count
    else:
        camera_superposition[i] = np.array([0, 0, 0])
    camera_superposition_track[i] = 1 if count > 0 else 0

#%%

def get_next_track_index(tracks, start_index):
    for i in range(start_index, len(tracks)):
        if tracks[i] == 1:
            return i
    return -1

# if the camera superposition track is 0, the camera superposition is set the average of the previous and next valid frame
for i in range(min_length):
    if camera_superposition_track[i] == 0:
        if INTERPOLATE_MISSING:
            next_track_index = get_next_track_index(camera_superposition_track, i + 1)
            camera_superposition[i] = (camera_superposition[i - 1] + camera_superposition[next_track_index]) / 2
        else:
            camera_superposition[i] = np.array([np.nan, np.nan, np.nan])


#%%
if not args.skip_plot:
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))

    for cam_id in camera_ids:
        positions = camera_room_positions[cam_id]
        axs[0].plot(timestamps, positions[:, 0], label=f'cam{cam_id} x')
        axs[1].plot(timestamps, positions[:, 1], label=f'cam{cam_id} y')
        axs[2].plot(timestamps, positions[:, 2], label=f'cam{cam_id} z')
        axs[3].plot(timestamps, camera_tracks[cam_id], label=f'cam{cam_id} track')

    axs[0].set_ylabel('X pos [m]')
    axs[1].set_ylabel('Y pos [m]')
    axs[2].set_ylabel('Z pos [m]')
    axs[3].set_ylabel('Has track')
    axs[3].set_xlabel('Time [s]')

    axs[0].plot(timestamps, camera_superposition[:, 0], label='super x', color='black', linestyle='--')
    axs[1].plot(timestamps, camera_superposition[:, 1], label='super y', color='black', linestyle='--')
    axs[2].plot(timestamps, camera_superposition[:, 2], label='super z', color='black', linestyle='--')
    axs[3].plot(timestamps, camera_superposition_track, label='super track', color='black', linestyle='--')

    for ax in axs:
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()

if ground_truth_output_folder and not os.path.exists(ground_truth_output_folder):
    os.makedirs(ground_truth_output_folder, exist_ok=True)
np.savez(os.path.join(ground_truth_output_folder, f"ground_truth_joint{joint_id}.npz"), trajectory=camera_superposition, timestamps=common_utc_timestamps)
