import cv2
import numpy as np
import glob
import os

class Calibration:
    def __init__(self, img_dir, grid_size, chess_size):
        '''
        初始化

        Parameters
        ----------
        img_dir : str
            图像的存储路径
            
        grid_size : int
            一个棋盘格的大小
            
        chess_size : (w, h)
            棋盘的宽和长分别有几个角点

        Returns
        -------
        None.

        '''
        #棋盘格的大小
        self.grid_size = grid_size
        #棋盘大小，即包括几行几列
        self.chess_size = chess_size
        w, h = chess_size
        #初始化世界坐标系点
        self.world_point = np.zeros((w * h, 3), np.float32)
        #根据棋盘格大小构造每一个世界坐标系的角点坐标
        self.world_point[:, :2] = np.mgrid[0:w*grid_size:grid_size, 0:h*grid_size:grid_size].T.reshape(-1, 2)
        #初始化世界坐标系，像素坐标系和图像路径
        self.world_points = []
        self.img_points = []
        self.img_dir = img_dir
        
        
    def chess_generator(self, save_dir, x_nums=14, y_nums=7):
        '''
        生成标定用棋盘格

        Parameters
        ----------
        save_dir : str
            棋盘图保存路径
            
        x_nums : int, optional
            生成棋盘的长 The default is 14.对应h=13
            
        y_nums : TYPE, optional
            生成棋盘的宽 The default is 7. 对应w=6

        Returns
        -------
        None.

        '''
        image = np.ones([1080, 1920, 3], np.uint8) * 255
        #x_nums = 14  #对应h=13                                           
        #y_nums = 7 #对应w=6                                
        square_pixel = 120   # 1080/9 = 120 pixels
        x0 = square_pixel
        y0 = square_pixel
        flag = -1                                        
        for i in range(y_nums):
            flag = 0 - flag
            for j in range(x_nums):
                if flag > 0:
                    color = [0,0,0]                         
                else:
                    color = [255,255,255]
                cv2.rectangle(image,(x0 + j*square_pixel,y0 + i*square_pixel),
                              (x0 + j*square_pixel+square_pixel,y0 + i*square_pixel+square_pixel),color,-1)
                flag = 0 - flag
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(save_dir+'/chess_map.bmp',image)
        print("done")
        
        
    def calibrate(self):
        '''
        相机标定

        Returns
        -------
        mat_intri : TYPE
            内参矩阵
            
        coff_dis : TYPE
            畸变系数，(k_1, k_2, p_1, p_2, k_3)
            
        vec_rot : TYPE
            外参：旋转向量
            
        vec_trans : TYPE
            外参：平移向量

        '''
        #图像路径
        self.images = glob.glob(self.img_dir + '/*.jpg')
        w, h = self.chess_size
        for idx, img_path in enumerate(self.images):
            #读取图像
            img = cv2.imread(img_path)
            #将图片缩小
            #img = cv2.resize(img,None,fx=0.4, fy=0.4, interpolation = cv2.INTER_CUBIC)
            #转为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #寻找角点
            ret, corners = cv2.findChessboardCorners(gray, (w,h), None)
            if ret:  
                #将角点精细化到亚像素
                #criteria:角点精准化迭代过程的终止条件
                #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
                #cv2.cornerSubPix(gray, corners, (11,11),(-1,-1),criteria)
                #向世界坐标系中增加点
                self.world_points.append(self.world_point)
                #向像素坐标系中增加角点
                self.img_points.append(corners)
                #在图像中标出角点
                cv2.drawChessboardCorners(img, (w,h), corners, ret)
                cv2.imshow('corners',img)
                cv2.waitKey(0)
                print(str(idx + 1) + " finish")
            else:
                print(str(idx + 1) + " wrong")
        cv2.destroyAllWindows()
        
        #求解摄像机的内在参数和外在参数。mat_intri 内参数矩阵，coff_dis 畸变系数，vec_rot 旋转向量，vec_trans 平移向量 
        ret, mat_intri, coff_dis, vec_rot, vec_trans = cv2.calibrateCamera(self.world_points, self.img_points, gray.shape[::-1], None, None)
        
        print ("ret: {}".format(ret))
        print ("intrinsic matrix: \n {}".format(mat_intri))
        # in the form of (k_1, k_2, p_1, p_2, k_3)
        print ("distortion cofficients(k_1, k_2, p_1, p_2, k_3): \n {}".format(coff_dis))
        print ("rotation vectors: \n {}".format(vec_rot))
        print ("translation vectors: \n {}".format(vec_trans))
        self.mat_intri = mat_intri
        self.coff_dis = coff_dis
        self.vec_rot = vec_rot
        self.vec_trans = vec_trans
        return mat_intri, coff_dis, vec_rot, vec_trans
    
    def undistort(self, save_dir):
        '''

        Parameters
        ----------
        save_dir : str
            去畸变图像保存路径

        Returns
        -------
        None.

        '''
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.mat_intri is None:
            self.calibrate_camera()
     

        for img_path in self.images:
            _, img_name = os.path.split(img_path)
            img = cv2.imread(img_path)
            #将图片缩小
            #img = cv2.resize(img,None,fx=0.4, fy=0.4, interpolation = cv2.INTER_CUBIC)
            h,  w = img.shape[:2]
            #去除畸变矫正后的鱼眼区域
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mat_intri, self.coff_dis, (w,h), 0, (w,h))
            dst = cv2.undistort(img, self.mat_intri, self.coff_dis, None, newcameramtx)
            # clip the image
            # x, y, w, h = roi
            # dst = dst[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(save_dir, img_name), dst)
        print("Dedistorted images have been saved to: {}".format(save_dir))
        
if __name__=='__main__':
    ca = Calibration('.', 33, (6,13))
    ca.chess_generator('./chessmap')
    ca.calibrate()
    ca.undistort('./undistort')
        

        
        
        
        
        