import cv2
import imutils
import numpy as np
import process_reps_video
import get_color
import pickle

def main():
    labeled_data = []

    video_names = ["testVideo2.mp4", "testVideo3.mp4", "testVideo4.mp4"]

    for video_name in video_names:
        first_time = True
        frame_num = 0

        print("Loading video:", video_name)
        cap = cv2.VideoCapture(video_name)

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

            img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_AREA)

            thresh = process_reps_video.color_mask(img, upper, lower)
            # cv2.imshow("thresh", thresh)
            # cv2.waitKey(0)
            # input()
            center = process_reps_video.get_location(thresh, frame_num)
            frame_num += 1
            labeled_data.append([img, center[0], center[1]])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    with open('labeled_data.pkl', 'wb') as f:
        print("Writing labeled data to file")
        print("Number of frames: ", len(labeled_data))
        pickle.dump(labeled_data, f)




if __name__ == "__main__":
    main()