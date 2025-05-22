import argparse
import pykinect_azure as pykinect
import pykinect_azure.k4a._k4a as k4a
import pykinect_azure.k4abt._k4abt as k4abt
import time
import cv2
import threading
import json
import numpy as np
import os
from aruco_detector import ArucoDetector
from screeninfo import get_monitors

if len(get_monitors()) > 1:
    monitor = get_monitors()[1]
else:
    monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height
window_size = 0

record = True
upside_down = False
data_folder = None
running = True

depth_video_width = 512
depth_video_height = 512

def parse_arguments():
    parser = argparse.ArgumentParser(description="Kinect Azure Multicam Body Tracking")
    parser.add_argument("--calib", action="store_true", help="Run in calibration mode")
    # TODO: light model stopped working after merging with main branch
    parser.add_argument("--lite", action="store_true", help="Use lite model for body tracking")
    parser.add_argument("--flip", action="store_true", help="Flip the camera upside down")
    parser.add_argument("--record", action="store_true", help="Record data to file")
    return parser.parse_args()

def start_camera(device_info):
    device = device_info['device']
    device.start_cameras(device_info['config'])
    print(
        f"Successfully started camera for device {device_info['index']} ({device_info['type']})")

def close_devices(devices):
    for device_info in devices:
        device_info['device'].close()

def process_camera_tracking(device_info, video_writer):
    frame_count = 0
    total_frame_count = 0
    start_time = time.time()
    global running

    if record:
        json_file_path = f"{data_folder}/track_data_cam{device_info['index']}.json"
        json_file = open(json_file_path, "w")
        json_file.write("[")

    window_name = f"Cam{device_info['index']}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, window_size * (device_info["index"]), 0)
    cv2.resizeWindow(window_name, window_size, window_size)

    while running:
        device = device_info['device']
        bodyTracker = device_info['bodyTracker']

        pre_time = time.time_ns()
        capture = device.update()
        utc_timestamp_us = (time.time_ns() + pre_time) // 2000
        device_timestamp_us = k4a.k4a_image_get_device_timestamp_usec(k4a.k4a_capture_get_depth_image(capture.handle()))

        ret_depth, depth_image = capture.get_colored_depth_image()
        if not ret_depth:
            continue


        body_frame = bodyTracker.update(device)

        _, body_image_color = body_frame.get_segmentation_image()

        combined_image = cv2.addWeighted(depth_image, 0.6, body_image_color, 0.4, 0)
        combined_image = body_frame.draw_bodies(combined_image)

        num_bodies = body_frame.get_num_bodies()
        if num_bodies > 0:
            joints = body_frame.json()[0]['skeleton']['joints']
        else:
            joints = ""

        if upside_down:
            combined_image = cv2.flip(combined_image, 0)

        if record:
            track_data = {
                "frame": total_frame_count,
                "utc_timestamp_us": utc_timestamp_us,
                "device_timestamp_us": device_timestamp_us,
                "num_bodies": num_bodies,
                "joints": joints}

            json.dump(track_data, json_file)
            json_file.write(",\n")
            json_file.flush()

            down_scaled_image = cv2.resize(combined_image, (depth_video_width, depth_video_height))
            video_writer.write(down_scaled_image)

        window_name = f"Cam{device_info['index']}"
        cv2.imshow(window_name, combined_image)

        frame_count += 1
        total_frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            actual_fps = frame_count / elapsed_time
            # print(f"Cam{device_info['index']} - Actual FPS: {actual_fps:.2f}")
            frame_count = 0
            start_time = time.time()

        key = cv2.waitKey(1)
        if key == ord('q'):
            running = False
            break

    # remove the last newline and comma
    if record:
        json_file.seek(json_file.tell() - 3, 0)
        json_file.truncate()
        json_file.write("]")
        json_file.close()

        video_writer.release()
    device.close()


def process_camera_calibration(device_info, aruco_detector: ArucoDetector):
    global running

    if record:
        json_file_path = f"{data_folder}/calibration_data_cam{device_info['index']}.json"
        json_file = open(json_file_path, "w")
        json_file.write("[")

    window_name = f"Cam{device_info['index']}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    row = device_info["index"] // 2
    col = device_info["index"] % 2
    cv2.moveWindow(window_name, col * window_size, row * window_size//2)
    cv2.resizeWindow(window_name, window_size, window_size//2)

    total_frame_count = 0

    while running:
        device = device_info['device']

        capture = device.update()
        ret_depth, depth_image = capture.get_transformed_depth_image()
        if not ret_depth:
            continue
        
        ret_color, color_image = capture.get_color_image()
        color_image, c0, x_vec, y_vec, z_vec = aruco_detector.detect(device.calibration, color_image, depth_image)

        ret_depth, colored_depth_image = capture.get_transformed_colored_depth_image()
        combined_image = cv2.addWeighted(color_image, 0.8, colored_depth_image, 0.2, 0)

        if record and c0 is not None and not np.isnan(c0[0]):
            calibration_data = {
                "frame": total_frame_count,
                "c0": c0.tolist(),
                "x_vec": x_vec.tolist(),
                "y_vec": y_vec.tolist(),
                "z_vec": z_vec.tolist()
            }

            json.dump(calibration_data, json_file)
            json_file.write(",\n")
            json_file.flush()

        if upside_down:
            combined_image = cv2.flip(combined_image, 0)

        window_name = f"Cam{device_info['index']}"
        cv2.imshow(window_name, combined_image)

        total_frame_count += 1

        key = cv2.waitKey(1)
        if key == ord('q'):
            running = False
            break

    if record:
        json_file.seek(json_file.tell() - 3, 0)
        json_file.truncate()
        json_file.write("]")
        json_file.close()

    device.close()

def main():
    args = parse_arguments()
    calibration = args.calib
    global upside_down, record
    upside_down = args.flip
    use_lite_model = args.lite
    record = args.record

    pykinect.initialize_libraries(track_body=not calibration)

    devices = []
    num_devices = pykinect.k4a_device_get_installed_count()
    if num_devices == 0:
        raise Exception("No Kinect devices found!")

    if record:
        if calibration:
            folders_path = "data/calibration"
            folder_prefix = "data/calibration/cal_"
        else:
            folders_path = "data/tracking"
            folder_prefix = "data/tracking/track_"

        global data_folder
        os.makedirs(folders_path, exist_ok=True)
        folders = os.listdir(folders_path)
        sorted_folders = sorted(folders, key=lambda x: int(x.split('_')[1]))
        highest_index = int(sorted_folders[-1].split('_')[1]) if sorted_folders else 0
        data_folder = f"{folder_prefix}{(highest_index + 1):04d}/"
        os.makedirs(data_folder, exist_ok=True)

    global window_size
    if not calibration:
        window_size = screen_width // num_devices
        window_size = min(window_size, screen_height)
    else:
        window_size = screen_width // 2

    if use_lite_model:
        fps = 15
        camera_fps = k4a.K4A_FRAMES_PER_SECOND_15
        model = k4abt.K4ABT_LITE_MODEL
    else:
        fps = 5
        camera_fps = k4a.K4A_FRAMES_PER_SECOND_5
        model = k4abt.K4ABT_DEFAULT_MODEL

    for i in range(num_devices):
        device = pykinect.Device(i)
        device_config, device_type = device.device_configinit()
        if calibration:
            device_config.depth_mode = k4a.K4A_DEPTH_MODE_NFOV_UNBINNED
            device_config.color_resolution = k4a.K4A_COLOR_RESOLUTION_3072P
            device_config.camera_fps = k4a.K4A_FRAMES_PER_SECOND_5
        else:
            device_config.depth_mode = k4a.K4A_DEPTH_MODE_WFOV_UNBINNED
            device_config.color_resolution = k4a.K4A_COLOR_RESOLUTION_720P
            device_config.camera_fps = camera_fps
        bodyTracker = None
        devices.append({
            'device': device,
            'bodyTracker': bodyTracker,
            'type': device_type,
            'config': device_config,
            'index': i,
            })

    master_devices = [d for d in devices if d['type'] == 'Master']
    sub_devices = [d for d in devices if d['type'] == 'Sub']
    stan_devices = [d for d in devices if d['type'] == 'Standalone']

    for device_info in stan_devices:
        start_camera(device_info)
    for device_info in sub_devices:
        start_camera(device_info)
    for device_info in master_devices:
        start_camera(device_info)

    if len(master_devices) == 1 and len(sub_devices) == 0:
        close_devices(devices)
        raise Exception(
            "NO Sub device detected but detected Master device, please check the sync cable!")

    elif len(master_devices) > 1:
        close_devices(devices)
        raise Exception(
            "The Master device cannot be more than one, please check the sync cable!")

    elif len(master_devices) == 0 and len(sub_devices) != 0:
        close_devices(devices)
        raise Exception(
            "NO Master device detected but detected Sub device, please check the sync cable!")

    tracker_config = pykinect.default_tracker_configuration
    if upside_down:
        tracker_config.sensor_orientation = k4abt.K4ABT_SENSOR_ORIENTATION_FLIP180
    else:
        tracker_config.sensor_orientation = k4abt.K4ABT_SENSOR_ORIENTATION_DEFAULT

    threads = []
    aruco_detector = ArucoDetector()

    for i in range(num_devices):
        if not calibration:
            devices[i]['bodyTracker'] = pykinect.start_body_tracker(calibration=devices[i]['device'].calibration, model_type=model)

            if record:
                video_writer = cv2.VideoWriter(
                    f'{data_folder}/track_video_cam{i}.mp4',
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (depth_video_width, depth_video_height)
                )
            else:
                video_writer = None

            thread = threading.Thread(target=process_camera_tracking, args=(devices[i], video_writer))
        else:
            thread = threading.Thread(target=process_camera_calibration, args=(devices[i], aruco_detector))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    close_devices(devices)


if __name__ == "__main__":
    main()