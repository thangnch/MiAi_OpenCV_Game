import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

cam = cv2.VideoCapture(1)

frame_width = 1280
num_ball = 10
move_speed = 30
ball_radius = 20
ball_color  = (255,0,0)
mStart = 49
mEnd = 68
laser_line_height = 30
current_mark = 0

ball_x = np.random.randint( 0, frame_width,num_ball)
print(ball_x)
ball_y = np.random.randint(-1000,0, num_ball)
print(ball_y)

detect_face = dlib.get_frontal_face_detector()
detect_mouth = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def mouth_aspect_ratio(lmouth):
    A = dist.euclidean(lmouth[2], lmouth[10])  # 51, 59
    B = dist.euclidean(lmouth[4], lmouth[8])  # 53, 57
    C = dist.euclidean(lmouth[0], lmouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar, lmouth[0]

def remove_add_ball(ball_x, ball_y, ball_to_remove):
    ball_x = np.delete(ball_x, ball_to_remove)
    ball_y = np.delete(ball_y, ball_to_remove)

    # Sinh ra bóng mới. Số bóng sỉnh ra = ball_to_remove
    new_ball_x = np.random.randint(0, frame_width, len(ball_to_remove[ball_to_remove]))
    new_ball_y = np.random.randint(-1000, 0, len(ball_to_remove[ball_to_remove]))

    ball_x = np.concatenate((ball_x, new_ball_x))
    ball_y = np.concatenate((ball_y, new_ball_y))

    return ball_x, ball_y

while True:
    ret, frame = cam.read()
    if ret:
        # Cho bóng rơi ngẫu nhiên
        move_y = np.random.randint( 0, move_speed, num_ball)
        ball_y = ball_y + move_y

        # Xoá các bóng vượt quá khung hình
        ball_to_remove = ball_y > frame.shape[0]
        ball_x, ball_y = remove_add_ball(ball_x, ball_y, ball_to_remove)



        for i in range(0, num_ball):
            cv2.circle(frame, (ball_x[i], ball_y[i]), ball_radius, ball_color, thickness=-1 )

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        rects = detect_face(gray, 0)

        for rect in rects:
            shape = detect_mouth(gray, rect)
            shape = face_utils.shape_to_np(shape)
            mouth = shape[mStart:mEnd]

            mouth_ratio, mouth_positon = mouth_aspect_ratio(mouth)

            if (mouth_ratio>0.75):
                laser_line_y = mouth_positon[1]

                cv2.rectangle(frame, (0, laser_line_y),
                              (frame_width, laser_line_y + laser_line_height), (0, 0, 255), thickness=-1)

                ball_to_remove = (ball_y >= laser_line_y) & (ball_y <= laser_line_y + laser_line_height)
                ball_x, ball_y = remove_add_ball(ball_x, ball_y, ball_to_remove)

                current_mark  =current_mark  + len(ball_to_remove[ball_to_remove])

        cv2.putText(frame, "Points:" + str(current_mark), (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Game", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()