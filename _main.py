# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 15:51:26 2023

@author: jomar
"""
# SPECIMENT ->

import json
import os
import cv2
import numpy as np
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_parameters(file_path):
    with open(file_path, 'r') as json_file:
        parameters = json.load(json_file)
    return parameters


def ensure_directories(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Check if a file exists
def file_exists(file_path):
    return os.path.exists(file_path) and os.path.isfile(file_path)


# Check a directory is empty
def is_directory_empty(directory_path):
    if not os.path.exists(directory_path):
        return False

    return len(os.listdir(directory_path)) == 0


def extract_frames(video_path, start_offset, output_folder, export_duration):
    ensure_directories(output_folder)
    
    ffmpeg_string = f'ffmpeg -loglevel warning -ss {start_offset} -t {export_duration} -i "{video_path}" {output_folder}frame_%d.jpg'
    print(ffmpeg_string)
    os.system(ffmpeg_string)


def get_first_and_last_frame_number(output_folder):
    frame_files = [file for file in os.listdir(output_folder) if file.startswith("frame_") and file.endswith(".jpg")]
    if not frame_files:
        return 0

    frame_numbers = [int(file.split("_")[1].split(".")[0]) for file in frame_files]
    return (min(frame_numbers), max(frame_numbers))


def crop_image(img, squares):
    if len(img.shape) == 2:  # Grayscale image
        height, width = img.shape
        channels = 1
    else:  # Color image
        height, width, channels = img.shape
        
    crop_left = img[squares["LU"][1]:squares["LD"][1], squares["LU"][0]:squares["LD"][0]]
    crop_right = img[squares["RU"][1]:squares["RD"][1], squares["RU"][0]:squares["RD"][0]]

    crop_img = np.zeros((height, width), np.uint8)
    crop_img[:, :] = 0

    crop_img[squares["LU"][1]:squares["LD"][1], squares["LU"][0]:squares["LD"][0]] = crop_left
    crop_img[squares["RU"][1]:squares["RD"][1], squares["RU"][0]:squares["RD"][0]] = crop_right

    #show_image(crop_img)
    return crop_img


def detect_circles(image, r_min, r_max, circ_param_1, circ_param_2):
    detected_circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 2 * r_max,
                                        param1=circ_param_1, param2=circ_param_2,
                                        minRadius=r_min, maxRadius=r_max)

    return detected_circles

        
def process_frame(img, frame_number, squares, circle_left, circle_right, r_min, r_max, circ_param_1, circ_param_2):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    # Try different block sizes and constants
    block_size = 15
    constant = 4
    
    imgf = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, block_size, constant)
    imgf = cv2.bitwise_not(cv2.Canny(imgf, 200, 1000))

    # Apply morphological operations (erosion and dilation) to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #imgf = cv2.morphologyEx(imgf, cv2.MORPH_OPEN, kernel)

    crop_img = crop_image(imgf, squares)

    detected_circles = detect_circles(crop_img, r_min, r_max, circ_param_1, circ_param_2)

    # Export the processed image with the circles detected
    export_processed_frame(imgf, squares, "export/frames_threshold", frame_number)

    return imgf, crop_img, detected_circles


def export_processed_frame(img, squares, output_folder, frame_number, sulfix = ""):
    # Create a copy of the crop_img to draw the squares without modifying the original image
    img_with_squares = img.copy()
    
    # Draw the squares in blue on the image with squares
    cv2.rectangle(img_with_squares, tuple(squares['LU']), tuple(squares['LD']), (128, 128, 128), 5)
    cv2.rectangle(img_with_squares, tuple(squares['RU']), tuple(squares['RD']), (128, 128, 128), 5)
        
    # Save the processed image with the squares and circles detected
    output_image_path = os.path.join(output_folder, f'processed_frame_{frame_number}{sulfix}.jpg')
    cv2.imwrite(output_image_path, img_with_squares)
    

def calculate_distance(detected_circles, squares):
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        x1, x2, y1, y2 = 0, 0, 0, 0
        for circle in detected_circles[0, :]:
            x, y, r = int(circle[0]), int(circle[1]), int(circle[2])

            draw = False
            if ((x > squares['LU'][0] and x < squares['LD'][0]) and
                    (y > squares['LU'][1] and y < squares['LD'][1])):
                circle_left = r
                x1, y1 = x, y
                draw = True

            if ((x > squares['RU'][0] and x < squares['RD'][0]) and
                    (y > squares['RU'][1] and y < squares['RD'][1])):
                circle_right = r
                x2, y2 = x, y
                draw = True

        if x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0:
            distance = round(math.sqrt(abs(x2 - x1) ** 2 + abs(y2 - y1) ** 2), 2)
            return circle_left, circle_right, distance
    return circle_left, circle_right, 0


# Function to display image for debugging
def show_image(img, window_name="Debug Image"):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def parse_arduino_file(arduino_file):
    df = pd.read_csv(arduino_file, sep="\t")
    return df 


def find_flash_in_arduino(df):
    flash_arduino_data = df[df['READ_ON'] == 0].reset_index(drop=True).head(1)
    return flash_arduino_data


# The temperature where the flash stops is the same frame as it happens
def calculate_temperature_time_lists(df, flash_ard, dist_list, frame_stop_flash):
    temperature_list = [0 for _ in dist_list]
    tempo_list = [0 for _ in dist_list]
    
    temperature_list[frame_stop_flash] = df['READ_INT_TEMP'][flash_ard]
    tempo_list[frame_stop_flash] = 0
    
    count = 0
    for i in range(frame_stop_flash - 1, -1, -1):
        count += 1
        num = int((frame_stop_flash - i + 6) / 12)
        temperature_list[i] = df['READ_INT_TEMP'][flash_ard - num]
        tempo_list[i] = -16.68333 * count

    count = 0
    for i in range(frame_stop_flash + 1, len(temperature_list)):
        count += 1
        num = int((i - frame_stop_flash + 6) / 12)
        temperature_list[i] = df['READ_INT_TEMP'][flash_ard + num]
        tempo_list[i] = round(16.68333 * count, 5)

    return temperature_list, tempo_list


# Find the arduino line of this frame, using the difference from this frame to the flash
def find_frame_arduino_line(df, frame_number, flash_arduino_data, frames_per_second = 60):
    # Find the time difference.
    # IMPORTANT -> The frame 1 is not the first line of arduino, this function sincronize it
    flash_frame = flash_arduino_data['READ_NUMBER'][0]
    flash_time = flash_arduino_data['READ_TIMER'][0]
    
    delta_time_millis = 1000 * ((flash_frame - frame_number) / frames_per_second)
    delta_time_millis = round(delta_time_millis)

    frame_time = flash_time - round(delta_time_millis)
    
    # Find the closest value in the 'READ_TIMER' column to frame_time
    closest_value = df.iloc[(df['READ_TIMER'] - frame_time).abs().idxmin()]
    closest_value_df = closest_value.to_frame().transpose()
    
    return closest_value_df

    
def export_dataframe(df):
    export_file = os.path.join('out.csv')
    df.to_csv(export_file, index=False)
    df = df.replace("", np.nan)


def load_dataframe(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    return df


def plot_graphs(df_export, video_path):
    fig = plt.figure(figsize=(16, 9))
    plt.plot(df_export['Temperature'], df_export['Distance'])
    plt.savefig(os.path.join(video_path, 'DILAT.jpg'), dpi=120)

    fig = plt.figure(figsize=(16, 9))
    plt.plot(df_export['Temperature'])
    plt.savefig(os.path.join(video_path, 'Temperature.jpg'), dpi=120)

    fig = plt.figure(figsize=(16, 9))
    plt.plot(df_export['Distance'])
    plt.savefig(os.path.join(video_path, 'Distance.jpg'), dpi=120)

    fig = plt.figure(figsize=(16, 9))
    plt.plot(df_export['Time'])
    plt.savefig(os.path.join(video_path, 'Time.jpg'), dpi=120)

    plt.close("all")

# EASY DEBUG - Save the dist_list to a text file using numpy
def save_dist_list_to_file(dist_list, file_path):
    np.savetxt(file_path, dist_list, fmt='%.2f')

# EASY DEBUG - Load the dist_list from the text file using numpy
def load_dist_list_from_file(file_path):
    dist_list = np.loadtxt(file_path)
    return dist_list


# Create an empty DataFrame with specified columns
def create_empty_dataframe():
    columns = ['Frame', 'Time', 'Temperature', 'Arduino_Line',
               'Distance', 'CircleLeft_X', 'CircleLeft_Y', 'CircleLeft_Radius',
               'CircleRight_X', 'CircleRight_Y', 'CircleRight_Radius']
    df = pd.DataFrame(columns=columns)
    return df

# Function to append parameters as the last row to the DataFrame
def append_parameters(df, parameters_list):
    df.loc[df.shape[0]] = parameters_list
    return df


# Function to draw the squares and circles on the frame
def draw_shapes_on_frame(frame, row):
    # Draw the left and right squares in blue
    cv2.rectangle(frame, tuple(squares['LU']), tuple(squares['LD']), (255, 0, 0), 5)
    cv2.rectangle(frame, tuple(squares['RU']), tuple(squares['RD']), (255, 0, 0), 5)

    # Draw the left and right circles as small 3px squares in red
    left_circle_x, left_circle_y, left_circle_radius = row['CircleLeft_X'], row['CircleLeft_Y'], row['CircleLeft_Radius']
    right_circle_x, right_circle_y, right_circle_radius = row['CircleRight_X'], row['CircleRight_Y'], row['CircleRight_Radius']
    cv2.rectangle(frame, (int(left_circle_x) - 1, int(left_circle_y) - 1), (int(left_circle_x) + 1, int(left_circle_y) + 1), (0, 0, 255), 10)
    cv2.rectangle(frame, (int(right_circle_x) - 1, int(right_circle_y) - 1), (int(right_circle_x) + 1, int(right_circle_y) + 1), (0, 0, 255), 10)

    # Draw the circle radius in green
    cv2.circle(frame, (int(left_circle_x), int(left_circle_y)), int(left_circle_radius), (0, 255, 0), 5)
    cv2.circle(frame, (int(right_circle_x), int(right_circle_y)), int(right_circle_radius), (0, 255, 0), 5)


def analyze_frames():
    x, y, r = 0, 0, 0
    circle_left, circle_right = 0, 0
    df_frame_anaysis = create_empty_dataframe()
    
    # Loop through each frame from first to last (inclusive)
    for frame_number in range(first_frame, last_frame + 1):
        # Print frame number every 50 frames, so I know how is the progress
        if not frame_number % 50:
            print(round((100*(frame_number-first_frame)/(last_frame + 1 - first_frame)),2), "%")

        # Find the data from arduino for this frame
        this_frame_arduino_data = find_frame_arduino_line(df_arduino, frame_number, stop_flash_temperature)
        this_frame_arduino_data = this_frame_arduino_data.reset_index(drop=True)
        
        # Initialize distance variable for the current frame
        distance = 0

        # Read the image frame for the current iteration
        img = cv2.imread(os.path.join(output_folder, f"frame_{frame_number}.jpg"), cv2.IMREAD_COLOR)
        
        # Process the frame to get important elements (imgf: thresholded image, crop_img: cropped image, detected_circles: circles found)
        imgf, crop_img, detected_circles = process_frame(img, frame_number, squares, circle_left, circle_right,
                                                         r_min, r_max, circ_param_1, circ_param_2)

        # Calculate the distance and update the circle_left and circle_right values
        circle_left, circle_right, distance = calculate_distance(detected_circles, squares)
        
        # Save the parametes: 'Frame', 'Time', 'Temperature', 'Arduino_Line',
        #                     'Distance', 'CircleLeft_X', 'CircleLeft_Y', 'CircleLeft_Radius',
        #                     'CircleRight_X', 'CircleRight_Y', 'CircleRight_Radius
        time =            this_frame_arduino_data['READ_TIMER'][0]
        temperature =     this_frame_arduino_data['READ_INT_TEMP'][0]
        arduino_line =    this_frame_arduino_data['READ_NUMBER'][0]
        distance =        distance
        circle_left_x =   detected_circles[0][0][0]
        circle_left_y =   detected_circles[0][0][1]
        circle_left_r =   detected_circles[0][0][2]
        circle_right_x =  detected_circles[0][1][0]
        circle_right_y =  detected_circles[0][1][1]
        circle_right_r =  detected_circles[0][1][2]
        parameters_list = [frame_number, time, temperature, arduino_line, distance, 
                           circle_left_x, circle_left_y, circle_left_r, 
                           circle_right_x, circle_right_y, circle_right_r]
        
        df_frame_anaysis = append_parameters(df_frame_anaysis, parameters_list)
        
    return df_frame_anaysis


def draw_over_frames():
    # Loop through each row of the DataFrame and draw shapes on each frame
    for _, row in df_frame_anaysis[0:50].iterrows():
        frame_number =  round(row['Frame'])
    
        # Load the frame image corresponding to the frame number (you need to adapt this part)
        frame_path = f'export/frames/frame_{frame_number}.jpg'
        frame = cv2.imread(frame_path)
    
        # Draw shapes on the frame
        draw_shapes_on_frame(frame, row)
    
        # Save the modified frame with drawn shapes (you can choose a different output path)
        output_frame_path = f'export/frames_draw/frame_{frame_number}.jpg'
        cv2.imwrite(output_frame_path, frame)

if __name__ == "__main__":
    #
    ### IMPORT DATA FROM EXTERNAL FILES
    #
        
    # Load the parameters about the video
    parameters_file = "data/parameters.json"
    parameters = read_parameters(parameters_file)
    
    # Create some variables with those parameters, just to easy access.
    video_path =                parameters["video_path"]
    output_folder =             parameters["output_folder"]
    start_offset =              parameters["start_offset"]
    export_duration =           parameters["export_duration"]
    frame_rate =                parameters["frame_rate"]
    
    # Read parameters from parameters_flash.json
    parameters_flash_file = "data/parameters_flash.json"
    parameters_flash = read_parameters(parameters_flash_file)
    
    # Create the variables from the Flash sintering process
    crop_squares =              parameters_flash["squares"]
    r_min =                     parameters_flash["rMin"]
    r_max =                     parameters_flash["rMax"]
    circ_param_1 =              parameters_flash["circ_param_1"]
    circ_param_2 =              parameters_flash["circ_param_2"]
    ard_read =                  parameters_flash["ard_read"]
    vid_frame =                 parameters_flash["vid_frame"]
    frame_stop_flash =          parameters_flash["frame_stop_flash"]
    squares =                   parameters_flash["squares"]
    
    # Read data created by the circuit monitoring the process
    arduino_file = "data/Arduino.txt"
    df_arduino = parse_arduino_file(arduino_file)
    
    # Find the frame where the flash finished
    stop_flash_temperature =    find_flash_in_arduino(df_arduino)
    
    #
    ### EXTRACT THE FRAMES AND GET INFORMATION ABOUT IT
    #
        
    # If the export frames directory is empty, then extract the frames
    if is_directory_empty(output_folder):
        extract_frames(video_path, start_offset, output_folder, export_duration)
    
    # Get the first and the last frames that were extracted
    first_frame, last_frame = get_first_and_last_frame_number(output_folder)
    print("Found frames between:", first_frame, "and", last_frame)
    
    #
    ### ANALYZE EACH FRAME
    #
    # If the analyze file exists, then just load
    if not file_exists("out.csv"):
        df_frame_anaysis = analyze_frames()
    else:
        df_frame_anaysis = load_dataframe('out.csv')
    
    #
    ### DRAW OVER THE FRAMES
    #
    draw_over_frames()
    
    
    
    #dist_list = load_dist_list_from_file('data/dist_list.txt')

    #print(dist_list)
    #export_dataframe(df_frame_anaysis)
    df_frame_anaysis = load_dataframe('out.csv')

    plot_graphs(df_frame_anaysis, "export")

    #temperature_list, tempo_list = calculate_temperature_time_lists(df_arduino, stop_flash_temperature, dist_list, frame_stop_flash)

