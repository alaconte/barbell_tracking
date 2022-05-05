import cv2
import imutils
import numpy as np
import copy
import get_color
import sys

from matplotlib import pyplot as plt


def color_mask(img, lower_bound, upper_bound):
    # lower_bound = np.array([0, 140, 0])  # [0, 180, 0]
    # upper_bound = np.array([130, 255, 100])  # [170, 255, 150]

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.inRange(rgb, lower_bound, upper_bound)


def get_location(img_masked, frame_num):
    contours = cv2.findContours(img_masked.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # get largest contour
    highest_area = 0
    idx = "none"

    for i, c in enumerate(contours):
        m = cv2.moments(c)
        if m['m00'] >= highest_area:
            highest_area = m['m00']
            idx = i

    M = cv2.moments(contours[idx])

    if M["m00"] == 0:
        print("frame num", frame_num)
        cv2.imshow("thresh", img_masked)
        cv2.waitKey(0)
        input()

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY, contours[i]


def draw_center(img, center):
    cX, cY, c = center

    cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
    cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(img, "center", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


# I couldn't use built in scipy functions to find local maxes
# because there were instances where the top of a rep would
# be the same value for multiple frames and these would not
# be detected as extrema. This function looks for the last
# value in a "plateau" like this and includes it as an extrema
def find_local_maxes(y_pos):
    idx = 0
    maxes = []
    while idx < len(y_pos) - 1:
        if y_pos[idx] < y_pos[idx + 1]:
            if idx + 2 < len(y_pos) and y_pos[idx + 2] < y_pos[idx + 1]:
                maxes.append(idx + 1)
                idx = idx + 2
            elif idx + 2 < len(y_pos) and y_pos[idx + 2] == y_pos[idx + 1]:
                left_val = y_pos[idx]
                tie_val = y_pos[idx + 1]
                idx += 2
                while idx + 1 < len(y_pos) and y_pos[idx + 1] == tie_val:
                    idx += 1
                if idx + 1 < len(y_pos) and y_pos[idx + 1] < tie_val:
                    maxes.append(idx + 1)
            else:
                idx += 1
        else:
            idx += 1
    return maxes


def find_local_mins(y_pos):
    idx = 0
    mins = [0]
    while idx < len(y_pos) - 1:
        if y_pos[idx] > y_pos[idx + 1]:
            if idx + 2 < len(y_pos) and y_pos[idx + 2] > y_pos[idx + 1]:
                mins.append(idx + 1)
                idx = idx + 2
            elif idx + 2 < len(y_pos) and y_pos[idx + 2] == y_pos[idx + 1]:
                left_val = y_pos[idx]
                tie_val = y_pos[idx + 1]
                idx += 2
                while idx + 1 < len(y_pos) and y_pos[idx + 1] == tie_val:
                    idx += 1
                if idx + 1 < len(y_pos) and y_pos[idx + 1] > tie_val:
                    mins.append(idx + 1)
            else:
                idx += 1
        else:
            idx += 1
    return mins


def split_reps(positions):
    debug = False

    reps = []
    keyframes = []
    current_idx = 0

    x_pos, y_pos = zip(*positions)
    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)

    max_y = np.amax(y_pos)
    min_y = np.amin(y_pos)
    range = max_y - min_y

    # bottom of a rep will be a local max since pixel values increase at bottom of image
    local_maxes = find_local_maxes(y_pos)
    local_mins = find_local_mins(y_pos)

    start_top = local_mins.pop(0)
    bottom = local_maxes.pop(0)

    if debug:
        print("local maxes", local_maxes)
        print("local mins", local_mins)
        print()
        print()

    counter = 1

    while (len(local_maxes) >= 1):
        if debug:
            print("start top:", start_top)
            print("bottom:", bottom)

        # make sure bottom is real bottom
        while len(local_maxes) > 0 and (y_pos[bottom] - min_y < range * 0.8 or local_maxes[0] - bottom < 10):
            bottom = local_maxes.pop(0)
            if debug:
                print("skipping bottom:", bottom)

        if len(local_maxes) < 1 and y_pos[bottom] * 1.2 < max_y:
            break

        # make sure top of rep is last local min before rep starts
        if len(local_mins) < 1 and y_pos[start_top] - min_y < 0.2 * range:
            break
        while len(local_mins) > 0 and local_mins[0] < bottom and y_pos[local_mins[0]] - min_y < 0.2 * range:
            if debug:
                print("skipping starting top")
            start_top = local_mins.pop(0)

        if len(local_mins) < 1:
            break

        end_top = local_mins.pop(0)

        # find ending top of rep
        while end_top < bottom or y_pos[end_top] - min_y > range * 0.2:
            if len(local_mins) < 1:
                break
            if debug:
                print("skipping ending top")
            end_top = local_mins.pop(0)

        if debug:
            print("rep", counter, " starting at idx: ", start_top, " ending at idx: ", end_top, "\n\n")
        reps.append((x_pos[start_top:end_top], y_pos[start_top:end_top]))
        keyframes.append(start_top)
        start_top = end_top
        if len(local_maxes) < 1:
            break
        bottom = local_maxes.pop(0)
        counter += 1

    if debug:
        plt.plot(y_pos)
        plt.xlabel("Frame number")
        plt.ylabel("Positions (pixels)")
        plt.title("Vertical position by frame")
        plt.show()

    return reps, keyframes


def main():
    first_time = True

    positions = []

    video_name = "testVideo3.mp4"

    if len(sys.argv) > 1:
        video_name = sys.argv[1]

    save_rep_videos = True
    full_length_vid = True

    print("Loading video")
    cap = cv2.VideoCapture(video_name)

    frame_num = 1

    while True:
        success, img = cap.read()
        if not success:
            if first_time:
                print("Error loading video")
                break
            else:
                print("Finished processing video")
                break

        # record data for first frame
        if first_time:
            print("Processing video")
            first_time = False
            first_img = img

            # get color to find
            pick_color = get_color.Pick_color()
            pick_color.user_choose_color(first_img)
            upper, lower = pick_color.set_thresholds()
            upper = np.array(upper)
            lower = np.array(lower)

        thresh = color_mask(img, upper, lower)
        # cv2.imshow("thresh", thresh)
        # cv2.waitKey(0)
        # input()
        center = get_location(thresh, frame_num)
        frame_num += 1
        positions.append((center[0], center[1]))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    reps, keyframes = split_reps(positions)

    print()

    # re-open video to get keyframes for drawing bar path
    cap = cv2.VideoCapture(video_name)

    for idx, rep in enumerate(reps):
        x_pos = rep[0]
        y_pos = rep[1]
        time = round(len(y_pos) * (1. / 30.), 2)
        print("Rep", idx + 1, "took:", time, "seconds")

        # draw bar path on image and save
        # get frame
        cap.set(1, keyframes[idx])
        success, start_frame = cap.read()
        # math to make gradient
        frames = len(x_pos)
        inc_per_frame = 255./frames
        height, width, depth = start_frame.shape

        # open video file for full length vid if necessary on first rep
        if idx == 0 and full_length_vid:
            full_vid = cv2.VideoWriter(video_name[0:len(video_name) - 4] + "_full_path.mp4",
                                       cv2.VideoWriter_fourcc(*'h264'), 30, (width, height))

        frame = copy.deepcopy(start_frame)
        for i in range(len(x_pos) - 1):
            x1 = x_pos[i]
            x2 = x_pos[i + 1]
            y1 = y_pos[i]
            y2 = y_pos[i + 1]
            # frame = cv2.circle(frame, (x, y), radius=0, color=(0, 0, 255), thickness=-1)
            red_val = round(255 - inc_per_frame * i)
            blue_val = 255 - red_val
            frame = cv2.line(frame, (x1, y1), (x2, y2), color=(blue_val, 0, red_val), thickness=3)

        x_text = round(width/25.)
        y_text = round((height/10.)*9)
        text_size = height*width*(2.5/(1920*1080))
        font = cv2.FONT_HERSHEY_DUPLEX
        frame = cv2.putText(frame, "Rep " + str(idx + 1) + " took " + str(time) + " seconds",
                            (x_text, y_text), font, text_size, color=(0, 0, 0), thickness=2)
        cv2.imwrite(video_name[0:len(video_name) - 4] +"_rep_" + str(idx + 1) + "_path.png",frame)

        # save videos of each rep if desired
        if save_rep_videos:
            rep_vid = cv2.VideoWriter(video_name[0:len(video_name) - 4] + "_rep_" + str(idx + 1) + "_path.mp4",
                                      cv2.VideoWriter_fourcc(*'h264'), 30, (width, height))
            vid_frame = cv2.putText(start_frame, "Rep " + str(idx + 1) + " took " + str(time) + " seconds",
                                    (x_text, y_text), font, text_size, color=(0, 0, 0), thickness=2)
            rep_vid.write(vid_frame)
            if full_length_vid:
                full_vid.write(vid_frame)
            for i in range(frames - 1):
                success, vid_frame = cap.read()
                vid_frame = cv2.putText(vid_frame, "Rep " + str(idx + 1) + " took " + str(time) + " seconds",
                                        (x_text, y_text), font, text_size, color=(0, 0, 0), thickness=2)
                for j in range(i - 1):
                    x1 = x_pos[j]
                    x2 = x_pos[j + 1]
                    y1 = y_pos[j]
                    y2 = y_pos[j + 1]
                    # frame = cv2.circle(frame, (x, y), radius=0, color=(0, 0, 255), thickness=-1)
                    red_val = round(255 - inc_per_frame * j)
                    blue_val = 255 - red_val
                    vid_frame = cv2.line(vid_frame, (x1, y1), (x2, y2), color=(blue_val, 0, red_val), thickness=3)
                rep_vid.write(vid_frame)
                if full_length_vid:
                    full_vid.write(vid_frame)
            rep_vid.release()
    if full_length_vid:
        full_vid.release()


    # x_pos = reps[0][0]
    # y_pos = reps[0][1]
    # print(len(x_pos))
    # print(len(y_pos))
    # plt.plot(x_pos)
    # plt.xlabel("Frame number")
    # plt.ylabel("Positions (pixels)")
    # plt.title("Horizontal position by frame")
    # plt.show()
    #
    # plt.plot(y_pos)
    # plt.xlabel("Frame number")
    # plt.ylabel("Positions (pixels)")
    # plt.title("Vertical position by frame")
    # plt.show()

    # display it
    # cv2.imshow("center", first_img)
    # cv2.waitKey(0)


if __name__ == "__main__":
    main()
