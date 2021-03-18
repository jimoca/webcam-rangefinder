import os
from cv2 import cv2 as cv 


def chess_snapshot(left_camera_id, right_camera_id, frameHeight, frameWidth):
        if not os.path.exists('./snapshot_640x240/'):
            os.makedirs('./snapshot_640x240/right/')
            os.makedirs('./snapshot_640x240/left/')
        cv.namedWindow("SnapShot")
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        leftCam = cv.VideoCapture(left_camera_id+cv.CAP_DSHOW)
        rightCam = cv.VideoCapture(right_camera_id+cv.CAP_DSHOW)
        
        leftCam.set(cv.CAP_PROP_FRAME_HEIGHT, frameHeight)
        leftCam.set(cv.CAP_PROP_FRAME_WIDTH, frameWidth)
        rightCam.set(cv.CAP_PROP_FRAME_HEIGHT, frameHeight)
        rightCam.set(cv.CAP_PROP_FRAME_WIDTH, frameWidth)
        if not (leftCam.isOpened() and rightCam.isOpened()):
            exit(1)
        img_num = 0
        while img_num < 60:
            retvalOfRight, rightFrame = rightCam.read()
            retvalOfLeft, leftFrame = leftCam.read()
            if not (retvalOfRight and retvalOfLeft):
                print("read fail ")
                break
            key = cv.waitKey(1)
            twoFrame = cv.hconcat([rightFrame, leftFrame])
            cv.imshow("SnapShot", twoFrame)
            if key & 0xFF == ord('q'):
                print("結束")
                break
            elif key & 0xFF == ord('s'):
                frameL = leftFrame
                frameR = rightFrame
            else:
                continue
            cv.imshow('imgR', frameR)
            cv.imshow('imgL', frameL)
            grayR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)
            grayL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)
           
            retR, cornersR = cv.findChessboardCorners(grayR, (9, 6), None)
            retL, cornersL = cv.findChessboardCorners(grayL, (9, 6),
                                                       None)  # Same with the left camera
            print(retL, retR)
            print(cornersL, cornersR)
            
            # 找角點
            if retR & retL:
                corners2R = cv.cornerSubPix(grayR, cornersR, (12, 12), (-1, -1), criteria)  # Refining the Position
                corners2L = cv.cornerSubPix(grayL, cornersL, (12, 12), (-1, -1), criteria)
                # 顯示角點
                cv.drawChessboardCorners(grayR, (9, 6), corners2R, retR)
                cv.drawChessboardCorners(grayL, (9, 6), corners2L, retL)
                cv.imshow('VideoR', grayR)
                cv.imshow('VideoL', grayL)
                cv.waitKey(0)
                cv.destroyAllWindows()
                cv.imwrite('./snapshot_640x240/right/' + 'right_' + str(img_num) + '.png', frameR)
                cv.imwrite('./snapshot_640x240/left/' + 'left_' + str(img_num) + '.png', frameL)
                print('Images ' + str(img_num) + ' is saved')
                img_num += 1
        print("完成")

if __name__ == '__main__':
    chess_snapshot(left_camera_id=1, right_camera_id=0, frameHeight=360, frameWidth=640)