import cv2
import numpy as np

class Pick_color():
    color = None

    def user_choose_color(self, img):
        color_explore = np.zeros((150, 150, 3), np.uint8)
        color_selected = np.zeros((150, 150, 3), np.uint8)
        color = "none"


        def select_color(event, x, y, flags, param):

            B = img[y, x][0]
            G = img[y, x][1]
            R = img[y, x][2]
            color_explore[:] = (B, G, R)

            if event == cv2.EVENT_LBUTTONDOWN:
                self.color = (R, G, B)

        # image window for sample image
        cv2.namedWindow('Select color to track')

        # mouse call back function declaration
        cv2.setMouseCallback('Select color to track', select_color)

        # while loop to live update
        while (self.color == None):
            cv2.imshow('Select color to track', img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()
        return

    def set_thresholds(self):
        lower = []
        upper = []
        for val in self.color:
            l = val - 40
            if l < 0:
                l = 0
            lower.append(l)
            u = val + 40
            if u > 255:
                u = 255
            upper.append(u)
        return lower, upper


def main():
    filename = "testPhoto2.jpg"
    pick_color = Pick_color()
    pick_color.user_choose_color(filename)
    print(pick_color.color)
    lower, upper = pick_color.set_thresholds()
    print("lower, ", lower, " upper, ", upper)


if __name__ == "__main__":
    main()