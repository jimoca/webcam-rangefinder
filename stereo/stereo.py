from cv2 import cv2 as cv
import numpy as np
import config
import time
import pyttsx3
import threading


class VoiceThread(threading.Thread):
    def __init__(self, rate=115, event=None):
        super().__init__()

        if event:
            setattr(self, event, threading.Event())

        self._cancel = threading.Event()
        self.rate = rate
        self.engine = None

        self._say = threading.Event()
        self._text_lock = threading.Lock()
        self._text = []

        self._is_alive = threading.Event()
        self._is_alive.set()
        self.start()

    def _init_engine(self, rate):
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)    
        engine.connect('finished-utterance', self._on_completed)
        engine.connect('started-word', self._on_cancel)
        return engine

    def say(self, text, stop=None):
        if self._is_alive.is_set():
            self._cancel.clear()

            if isinstance(text, str):
                text = [(text, stop)]

            if isinstance(text, (list, tuple)):
                for t in text:
                    if isinstance(t, str):
                        t = t, None

                    with self._text_lock:
                        self._text.append(t)

                    self._say.set()

    def cancel(self):
        self._cancel.set()

    def _on_cancel(self, name, location, length):
        if self._cancel.is_set():
            self.stop()

    def stop(self):        
        self.engine.stop()
        time.sleep(0.5)
        self.engine.endLoop()

    def _on_completed(self, name, completed):
        if completed:
            self.engine.endLoop()
            self.on_finished_utterance(name, completed)

    def on_finished_utterance(self, name, completed):
        pass

    def terminate(self):
        self._is_alive.clear()
        self._cancel.set()
        self.join()

    def run(self):
        self.engine = engine = self._init_engine(self.rate)
        while self._is_alive.is_set():
            while self._say.wait(0.1):
                self._say.clear()

                while not self._cancel.is_set() and len(self._text):
                    with self._text_lock:
                        engine.say(*self._text.pop(0))
                    engine.startLoop()






class Voice(VoiceThread):
    def __init__(self):
        self.completed = None
        super().__init__(rate=200, event='completed')


    def on_finished_utterance(self, name, completed):
        # print('speak finish')
        self.completed.set()
        
    


class Stereo:
    def __init__(self, left_camera_id, right_camera_id, frameHeight, frameWidth):
        self.left_camera_id = left_camera_id
        self.right_camera_id = right_camera_id
        self.frameHeight = frameHeight
        self.frameWidth = frameWidth
        self.filter_disp = None
        self.sentences = []
    
    def stereo_sgbm(self):

        voice = Voice()
        voice.completed.set()
        

        Left_Stereo_Map = (config.left_map1, config.left_map2)
        Right_Stereo_Map = (config.right_map1, config.right_map2)
        
        
        cv.namedWindow("Two Camera")
        # cv.namedWindow('depth')
        # cv.createTrackbar("windowsize", "depth", 1, 25, lambda x: None)
        # cv.createTrackbar("max disp", "depth", 247, 256, lambda x: None)
        leftCam = cv.VideoCapture(self.left_camera_id+cv.CAP_DSHOW)
        rightCam = cv.VideoCapture(self.right_camera_id+cv.CAP_DSHOW)

        leftCam.set(cv.CAP_PROP_FRAME_HEIGHT, self.frameHeight)
        leftCam.set(cv.CAP_PROP_FRAME_WIDTH, self.frameWidth)
        rightCam.set(cv.CAP_PROP_FRAME_HEIGHT, self.frameHeight)
        rightCam.set(cv.CAP_PROP_FRAME_WIDTH, self.frameWidth)
        
        # leftCam.set(cv.CAP_PROP_FPS, 3)
        # rightCam.set(cv.CAP_PROP_FPS, 3)

        # leftCam.set(cv.CAP_PROP_BUFFERSIZE, 3)
        _, fl = leftCam.read()
        # window_size = cv.getTrackbarPos("windowsize", "depth")
        window_size = 1
        # max_disp = cv.getTrackbarPos("max disp", "depth")
        max_disp = 247
        min_disp = 0
        num_disp = max_disp - min_disp
        print(len(fl.shape))
        p1_var = 8*len(fl.shape)*window_size*window_size
        p2_var = 32*len(fl.shape)*window_size*window_size
        stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                      numDisparities=num_disp,
                                      blockSize=window_size,
                                      uniquenessRatio=10,
                                      speckleWindowSize=100,
                                      speckleRange=1,
                                      disp12MaxDiff=10,
                                      P1=p1_var,
                                      P2=p2_var)
        # Used for the filtered image
        # Create another stereo for right this time
        stereoR = cv.ximgproc.createRightMatcher(stereo)

        # WLS FILTER Parameters
        lmbda = 80000
        sigma = 2.0

        wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)
        shot = True
        if not (leftCam.isOpened() and rightCam.isOpened()):
            exit(1)
        while True:
            retvalOfRight, rightFrame = rightCam.read()
            retvalOfLeft, leftFrame = leftCam.read()
            if not (retvalOfRight and retvalOfLeft):
                print("read fail")
                break
            key = cv.waitKey(1)
            twoFrame = cv.hconcat([rightFrame, leftFrame])
            cv.imshow("Two Camera", twoFrame)
            if key & 0xFF == ord('q'):
                print("結束")
                break
            # elif key & 0xFF == ord('s'):
            elif shot:
                frameL = leftFrame
                frameR = rightFrame
                shot = False
            else:
                time.sleep(0.1)
                shot = True
                continue

            
            remapped_left_side = cv.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv.INTER_LANCZOS4,
                                 cv.BORDER_CONSTANT,
                                 0)
            remapped_right_side = cv.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv.INTER_LANCZOS4,
                                  cv.BORDER_CONSTANT, 0)



            grayR = cv.cvtColor(remapped_right_side, cv.COLOR_BGR2GRAY)
            grayL = cv.cvtColor(remapped_left_side, cv.COLOR_BGR2GRAY)
            grayR = cv.equalizeHist(grayR)
            grayL = cv.equalizeHist(grayL)

            # cv.imshow('grayR', grayR)
            # cv.imshow('grayL', grayR)

            disp = stereo.compute(grayL, grayR)
            dispL = np.int16(disp)

            dispR = stereoR.compute(grayR, grayL)

            dispR = np.int16(dispR)
            # cv.imshow('dispR', dispR)
            filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
            # cv.imshow('filteredImg', filteredImg)

            filteredImg = cv.normalize(src=filteredImg, dst=filteredImg,
                                       beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
            filteredImg = np.uint8(filteredImg)


            # contours, hierarchy = cv.findContours(filteredImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE )
            # print(len(contours[0])) 
            # cv.drawContours(grayL, [cnt], 0, (0,255,0), 3)
            # cv.drawContours(filteredImg, contours, 1, (0,0,255), 20)


            self.filter_disp = (
                (filteredImg.astype(np.float32) / 16) - min_disp) / num_disp
            # filt_Color = cv.applyColorMap(filteredImg, cv.COLORMAP_RAINBOW )
            # cv.imshow('depth', filt_Color)

            cv.imshow('depth', filteredImg)
            left = cv.line(
                remapped_left_side, (0, 0), (0, self.frameHeight), (0, 0, 0), 1)
            
            self.detect()
            

            if len(self.sentences) == 1:
                if voice.completed.is_set():
                    voice.completed.clear()
                    voice.say(self.sentences.pop(len(self.sentences)-1))
            
            
            cv.imshow('calc', left)
            
        rightCam.release()
        leftCam.release()
        cv.destroyAllWindows()
        voice.terminate()

    def calc(self, x, y):
        avg = 0
        for u in range(-1, 2):
            for v in range(-1, 2):
                avg += self.filter_disp[y + u, x + v]
        avg = avg / 9
        d = -15137 * avg ** 3 + 4620.2 * avg ** 2 - 494.75 * avg + 21.176
        d = np.around(d * 10)
        
        return d
    

    def detect(self): 
        obstacle = False
        for i in range(200, self.frameWidth-50, 100):
            for j in range(100, self.frameHeight-100, 100):
                if(not(obstacle)):
                    d=self.calc(x=i, y=j)
                    d = round(round(d,-1)) -30
                    # print(d)
                    if(d<50):
                        print('x: ', i, 'y: ', j)
                        sentence = '距離約' + str(d) + 'cm'
                        if(len(self.sentences) < 1):
                            self.sentences.append(sentence)
                        else:
                            self.sentences[0] = sentence
                        print(self.sentences)
                        obstacle=True
                else :
                    break
        obstacle=False
            


if __name__ == '__main__':
    sgbm = Stereo(left_camera_id=0, right_camera_id=1, frameHeight=360,
                      frameWidth=640)

    sgbm.stereo_sgbm()
