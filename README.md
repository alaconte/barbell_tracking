# barbell_tracking
## Purpose
This is a project with the goal of tracking a barbell in a video to provide feedback on rep times and bar path.
## Use/current limitations
This project currently relies on tracking a colored marker placed on the end of the barbell. This can be any color as long as it can be easily distinguished from the other elements in the video. I suggest using a bright green, and this can be anything from colored duct tape to a sheet of paper.

To run the program, first install all of the required libraries (opencv, numpy, imutils...) and run the script process_reps_video.py in the same directory as your video with the video's filename as an argument.
EX:
```
python process_reps_video.py test_video.mp4
```
A window should pop up allowing you to select your marker in the frame, and then after processing videos and still frames with your bar path and time for each rep should be created and saved in the directory with the script
## Example output
Examples of videos that might be output after running this script can be found here: https://drive.google.com/drive/folders/1o6j6dSNbDBP2Ge2x1BgUkmyc3ywneKST?usp=sharing
