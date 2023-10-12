# %%
import cv2
from render.renderer import *
import numpy as np
import matplotlib.pyplot as plt
import time
import pyrr
import open3d as o3d
from scipy.interpolate import interp1d
import math
from PSO.PSO import PSO

class ProcessHelper:

    # 给度数，返回R矩阵
    def GetR(self,rx,ry,rz)->np.ndarray:
        rx = np.array(pyrr.Matrix44.from_x_rotation(rx/180*3.14159))[:3,:3]
        ry = np.array(pyrr.Matrix44.from_y_rotation(ry/180*3.14159))[:3,:3]
        rz = np.array(pyrr.Matrix44.from_z_rotation(rz/180*3.14159))[:3,:3]
        tmp = rz.dot(ry).dot(rx)
        return tmp

    # 给定xyzrxryrz，返回
    def compute_pose(self,z,rx,ry,rz=0,x=0,y=0):
        R = self.GetR(rx,ry,rz)
        PC1O = np.identity(4)
        PC1O[:3,:3]=R
        PC1O[0,3]=x
        PC1O[1,3]=y
        PC1O[2,3]=z
        return PC1O
    
    def getRxRy(self,pts):
        rxry = []
        for v in pts:
            x,y,z = v[0],v[1],v[2]
            z_ = np.array([-x,-y,-z])
            y__ = np.array([0,-1,0])
            x_ = np.cross(y__,z_)
            y_ = np.cross(z_,x_)
            x_ = x_/np.linalg.norm(x_)
            y_ = y_/np.linalg.norm(y_)
            z_ = z_/np.linalg.norm(z_)
            R = np.eye(3)
            R[:3,0] = x_.T
            R[:3,1] = y_.T
            R[:3,2] = z_.T
            R=R.T
            eulers = self.rotationMatrixToEulerAngles(R)*180/np.pi
            rxry.append([eulers[0],eulers[1]])
        return np.array(rxry[1:-1])
    
    def fibonacci_sphere(self,samples=100):
        points = []
        phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            points.append([x, y, z])
        return points

    # 对轮廓投影到球面上之后的点云做差值，按照new_distance作为新的3d点之间的间距
    def cubic_spline_interpolation(self,points, new_distance):
        pts = points.T
        # 计算每相邻两个点之间的距离
        distances = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        # 计算每个点在曲线上的累积距离
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        # 设定新的间距
        new_spacing = new_distance
        # 计算新的累积距离
        new_cumulative_distances = np.arange(0, cumulative_distances[-1], new_spacing)
        # 使用线性插值函数生成新的点序列
        interp_func = interp1d(cumulative_distances, pts, kind='linear', axis=0)
        new_points = interp_func(new_cumulative_distances)
        return new_points.T,distances

    # kernel
    # 给一张silhouette图，输出其contour在球面上的点云
    def getPtsOnSphere(self,img,new_distance,K,K_1,IF_RESAMPLE = True):
        try:
            img1 = img.copy()
            contours, hierarchy = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            mx = 0
            for contour in contours:
                if contour.shape[0] > mx:
                    pts1 = contour[:,0,:]
                    mx = contour.shape[0]
                    c=contour

            ones = np.ones(len(pts1)).reshape(-1, 1)
            pts1 = np.hstack((pts1, ones))
            pts1 = K_1@pts1.T
            pts1_len = np.linalg.norm(pts1,axis=0)
            pts1 = pts1/pts1_len[np.newaxis,:]
            x, y, w, h = cv2.boundingRect(c)
            if IF_RESAMPLE:
                npts,distances = self.cubic_spline_interpolation(pts1,new_distance)
            else:
                npts = pts1
            l1=0
            return npts,(x,y,w,h),l1
        except:
            return -1

    # kernel
    def icp_registration(self,source, target, threshold=0.02, max_iterations=2000,init_pose = None):
        if init_pose is not None:
            pose_init = init_pose
        else:
            pose_init = np.identity(4)
        # 使用ICP匹配点云
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, threshold, pose_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria( max_iteration=max_iterations, relative_fitness = 1e-25, relative_rmse = 1e-25),)
        return icp_coarse.transformation, icp_coarse

    def union_roi(self,bx1,bx2):
        x1,y1,w1,h1 = bx1
        x2,y2,w2,h2 = bx2
        x = min(x1,x2)
        y = min(y1,y2)
        x_ = max(x1+w1,x2+w2)
        y_ = max(y1+h1,y2+h2)
        return (x,y,x_-x,y_-y)

    # 灰度图
    def croped_img_using_bx(self,img,bx):
        return img[bx[1]:bx[1]+bx[3],bx[0]:bx[0]+bx[2]]

    # 计算两张silhouette图片在单位球面上的IOU
    def iou(self,m1, m2,scale_mp):
        intersection = np.logical_and(m1, m2)
        union = np.logical_or(m1, m2)
        iou_score = np.sum(scale_mp[intersection]) / np.sum(scale_mp[union])
        return iou_score

    def visualize_registration(self,source, target, transformation):
        # 将源点云应用变换矩阵
        source_ = o3d.geometry.PointCloud(source)
        if transformation is not None:
            source_.transform(transformation)

        # 设置点云颜色
        source_.paint_uniform_color([0, 0, 1])  
        target.paint_uniform_color([0, 1, 0])

        # 可视化匹配结果
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(source_)
        vis.add_geometry(target)

        render_option = vis.get_render_option()
        render_option.background_color = [0.4,0.4,0.4]
        vis.run()
        vis.destroy_window()


    # kernel
    def icp_rotation_PaI(self,pts1,pcd1,l1,img1,bx1,img2,K,K_1,scale_map = None):
        try:
            if scale_map is None:
                scale_map = np.ones_like(img1)
            debug = False
            if debug:
                s = time.time()
            pts2,bx2,l2 = self.getPtsOnSphere(img2,0,K,K_1,False)
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(pts2.T)
            s = time.time()

            ct1,_ = self.compute_center_and_main_direction(pts1,True)
            ct2,_ = self.compute_center_and_main_direction(pts2,True)
            R1 = self.compute_rotation_between_two_dirs(ct1,ct2).T # 质心重合用的R

            pose_ = None
            mn_rmse = 1e100
            for angle in range(0,360,180):
                R2 = self.compute_rotation_around_axis(ct2,angle)
                R = R2@R1
                init_pose = np.identity(4)
                init_pose[:3,:3] = R
                pose,_ = self.icp_registration(pcd2,pcd1,5000,200,init_pose=init_pose)
                pose = np.linalg.inv(pose)
                if _.inlier_rmse<mn_rmse:
                    mn_rmse = _.inlier_rmse
                    pose_ = pose
            R_ = pose_[:3,:3] # !!!这里的R_==R.T
            img = cv2.warpPerspective(img1,K@R_@K_1,(img1.shape[1],img1.shape[0]))
            
            # bx1和bx2求并
            bx = self.union_roi(bx1,bx2)
            iou_r = self.iou(
                self.croped_img_using_bx(img,bx),
                self.croped_img_using_bx(img2,bx),
                self.croped_img_using_bx(scale_map,bx),)
            return R_,iou_r
        except:
            return 0,0

    # 纯旋转icp
    # 计算同一个center的两个dr的角度变换矩阵(应该是绕着center的旋转矩阵，从dr1转向dr2)
    def compute_rotation_between_main_directions(self,center,dr1,dr2):
        # dr1,dr2投影到center的垂直平面上,也就是去掉center方向的投影
        ## 计算投影向量，并各自减去
        d1 = (np.dot(dr1,center)/np.dot(center,center))*center
        d2 = (np.dot(dr2,center)/np.dot(center,center))*center
        dir1 = dr1
        dir2 = dr2
        ## 计算dir1转向dir2的角度(绕center方向)
        return self.compute_rotation_between_two_dirs(dir1,dir2).T

    # kernel, 暂未实现，容易局部最优
    # 计算球面上的轮廓点云的质心和主方向
    def compute_center_and_main_direction(self,points,only_center=False):
        # 计算点云的中心
        centroid = np.mean(points, axis=1)
        if only_center:
            return centroid,0

        # 计算点云相对于中心的协方差矩阵
        centered_pts = points-centroid.reshape(-1,1)
        tcm = np.mean(centered_pts**3,axis=1) # 三阶中心矩
        covariance_matrix = np.cov(points-centroid.reshape(-1,1))

        # 使用numpy的linalg.eig函数计算协方差矩阵的特征向量和特征值
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # 将特征向量按特征值从大到小排序
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 主方向是特征向量矩阵的第一列
        main_direction = eigenvectors[:, 0]

        # 用偏度修正主方向的真方向
        if np.dot(main_direction,tcm) < 0:
            main_direction = -main_direction

        return centroid,main_direction
    
    def compute_rotation_between_two_dirs(self,dir1,dir2):
        t_target = dir1
        t_source = dir2
        theta = np.arccos(np.dot(t_source,t_target)/(np.linalg.norm(t_source)*np.linalg.norm(t_target)))
        rvec = np.cross(t_source,t_target)
        rvec = rvec/np.linalg.norm(rvec)*theta
        if theta != 0.0:
            R, _ = cv2.Rodrigues(rvec)
        else:
            R = np.identity(3)
        return R
        
    def compute_rotation_around_axis(self,axis,theta):
        axis = axis/np.linalg.norm(axis)*theta/180*np.pi
        if theta != 0.0:
            R, _ = cv2.Rodrigues(axis)
        else:
            R = np.identity(3)
        return R

    def rotationMatrixToEulerAngles(self,R) :
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])

    # kernel
    # 输入图像大小和对应相机的K值，求每个像素面积映射到球面上的面积大小
    # img_shape: (H,W)
    def create_scale_map(self,img_shape,cam_K):
        cam_K_1 = np.linalg.inv(cam_K)
        def f(u,v): # 返回cos^4 theta
            c = np.concatenate([u,v,np.ones(u.shape)])
            vec = c.reshape((3,-1))
            vec = cam_K_1@vec
            vec = vec[:2,:]
            tan_value = np.linalg.norm(vec,axis=0)
            cos_2_value = 1/(1+tan_value**2)
            return cos_2_value**1.5
        i,j = np.meshgrid(range(img_shape[1]),range(img_shape[0]))
        result = f(i.flatten(),j.flatten())
        return result.reshape(img_shape[0],-1)

    # kernel
    # 将PCO转换成rx ry z
    def pose_to_zrxry(self,PCO):
        # 先通过相机纯旋转(任意一个即可)，把物体挪移到z轴上，此时即可获得rx ry z，rz是不要的
        ## 先找到z值，然后两个向量确定一个旋转向量，然后转换为旋转矩阵乘到PCO上即可
        z = np.linalg.norm(PCO[:3,3])
        t_target = np.array([0,0,z])
        t_source = PCO[:3,3].T
        theta = np.arccos(np.dot(t_source,t_target)/(np.linalg.norm(t_source)*np.linalg.norm(t_target)))
        rvec = np.cross(t_source,t_target)
        rvec = rvec/np.linalg.norm(rvec)*theta
        if theta != 0.0:
            R, _ = cv2.Rodrigues(rvec)
            p = np.identity(4)
            p[:3,:3] = R
            PCO = p@PCO # 这个PCO的z轴是穿过物体坐标系原点的
        rs = self.rotationMatrixToEulerAngles(PCO[:3,:3])*180/np.pi
        return z,rs[0],rs[1]
    
    # 缩放K用于加速
    def get_scaled_K(self,K,ds_scale):
        res_K = K.copy()
        res_K[0,0]=res_K[0,0]//ds_scale
        res_K[1,1]=res_K[1,1]//ds_scale
        res_K[0,2]=res_K[0,2]//ds_scale
        res_K[1,2]=res_K[1,2]//ds_scale
        return res_K

    def gray_to_color(self,gray_image):
        height, width = gray_image.shape
        color_image = np.zeros((height, width, 3), dtype=np.uint8)
        gray_image = gray_image * 255
        color_image[:, :, 0] = gray_image.astype(np.uint8)
        color_image[:, :, 2] = (255 - gray_image).astype(np.uint8)

        return color_image


class Process:
    def __init__(self,image_size,K1,ds_scale = 1) -> None:
        self.helper = ProcessHelper()
        self.ori_image_size = image_size
        self.ori_K1 = K1
        self.ori_K1_1 = np.linalg.inv(K1)
        self.model_fn = ""
        self._set_ds_scale(ds_scale=ds_scale)
        self.ds_scale = ds_scale
        self.set_args()
    
    # 设置降采样系数，以及连带的一些更新
    def _set_ds_scale(self,ds_scale=1):
        self.ds_scale = ds_scale
        self.image_size = (self.ori_image_size[0]//ds_scale,self.ori_image_size[1]//ds_scale)
        self.renderer = create_renderer(*(self.image_size),'vispy','mask','phong') # 已经经过魔改，现在只出mask，这些光照模型是没有用上的
        if self.model_fn != "":
            self.set_model(self.model_fn)
        self.K1 = self.helper.get_scaled_K(self.ori_K1,ds_scale)
        self.K1_1 = np.linalg.inv(self.K1)
        self.scale_map1 = self.helper.create_scale_map((int(self.image_size[1]),int(self.image_size[0])),self.K1)
        
    # 有关精度的参数设置
    def set_args(self,corse_search_space=(1000,1000,1000),corse_voxel_size=5,fine_search_space=(150,150,150),fine_voxel_size=1.5,top_n=4):
        self.corse_search_space = corse_search_space # 粗重建时在第一个相机下建立的体素空间大小
        self.corse_voxel_size = corse_voxel_size # 粗粒度重建时用的体素大小，这个值必须小于要找的物体的直径（以后可以根据模型的大小，自动化计算
        self.fine_search_space = fine_search_space # 细粒度重建时在第一个相机下建立的体素空间大小
        self.fine_voxel_size = fine_voxel_size # 细粒度重建时用的体素大小，越小越准
        self.top_n = top_n # model向sc结果icp对齐时的输入候选数目，越大运行越慢，但越准。对于不太对称的物体而言，3~4的足够

    def set_model(self,fn):
        self.model_fn = fn
        self.model_pcd = o3d.io.read_point_cloud(fn) # 模型点云
        self.renderer.add_object(0,fn) # 给renderer设置模型

    # inv False时表示第一个参数为POC
    # id 表示是用哪个相机渲染图形,0表示第一个，1表示第二个相机。第一个相机是主相机
    def render_silhouette(self,PCO,id=0,inv=False):
        if inv:
            PCO = np.linalg.inv(PCO)
        if id == 0:
            K = self.K1
        elif id == 1:
            K = self.K2
        res = self.renderer.render_object(0,PCO[:3,:3],PCO[:3,3].T,K[0,0],K[1,1],K[0,2],K[1,2])
        grey = cv2.cvtColor(res['mask'], cv2.COLOR_RGB2GRAY)
        return grey

    # 说是rgb，其实最后返回的是灰度图，为了方便计算，效果不好的话再改成真rgb吧
    def render_rgb(self,PCO,id=0,inv=False):
        if inv:
            PCO = np.linalg.inv(PCO)
        if id == 0:
            K = self.K1
        elif id == 1:
            K = self.K2
        res = self.renderer_rgb.render_object(0,PCO[:3,:3],PCO[:3,3].T,K[0,0],K[1,1],K[0,2],K[1,2])
        grey = cv2.cvtColor(res['rgb'], cv2.COLOR_RGB2GRAY)
        return grey
    
    def getPtsOnSphere(self,id=0):
        if id==0:
            img,K,K_1 = self.img1,self.K1,self.K1_1
        elif id==1:
            img,K,K_1 = self.img2,self.K2,self.K2_1
        return self.helper.getPtsOnSphere(img,0.01,K,K_1)

    # 设置参考图像
    # img1是silhouette图，单通道
    def set_ref(self,img1):
        img1 = cv2.resize(img1,(img1.shape[1]//self.ds_scale,img1.shape[0]//self.ds_scale))
        self.img1 = img1
        self.pts1,self.bx1,self.l1 = self.helper.getPtsOnSphere(img1,0.0005,self.K1,self.K1_1,True) # pts为3XN
        self.pcd1 = o3d.geometry.PointCloud()
        self.pcd1.points = o3d.utility.Vector3dVector(self.pts1.T)
    
    def PC1O2PC2O(self,PC1O,PC1C2):
        return np.linalg.inv(PC1C2)@PC1O

    def icp_rotation_PaI(self,img_cur,id=0,show = False,only_one = False):
        if id==0:
            pts,pcd,l,img_ref,bx,K,K_1,scale_map = self.pts1,self.pcd1,self.l1,self.img1,self.bx1,self.K1,self.K1_1,self.scale_map1
        elif id==1:
            pts,pcd,l,img_ref,bx,K,K_1,scale_map = self.pts2,self.pcd2,self.l2,self.img2,self.bx2,self.K2,self.K2_1,self.scale_map2
        return self.helper.icp_rotation_PaI(pts,pcd,l,img_ref,bx,img_cur,K,K_1,scale_map = scale_map)
    


    def get_value(self,z,rx,ry):
        try:
            PC1O = self.helper.compute_pose(z,rx,ry)
            img1 = self.render_silhouette(PC1O,id=0) # pose1的未矫正图像，也就是当前猜测视角下的图像，不考虑旋转
            R1,score1 = self.icp_rotation_PaI(img1,id=0,show=False,only_one = True)


            # if score1>0.8:
                # # 渲染一张rgb图，并且和img2比较，最后和iou的值共同返回一个score
                # rgb_rendered = self.render_rgb(PC1O,id=0)
                # H = self.K1@R1@self.K1_1
                # # rgb_warped = cv2.warpPerspective(rgb_rendered,H,(img1.shape[1],img1.shape[0]))
                # ## rgb_warped与rgb比较,在img1(mask)范围内
                # rgb_rendered[img1==0] = 0
                # img2 = np.copy(self.img2)
                # img2[self.img1==0] = 0
                # s,_ = structural_similarity(img2,rgb_rendered,full=True)
                # cv2.imwrite("test1.png",img2)
                # cv2.imwrite("test2.png",rgb_rendered)
                # cv2.imwrite("test3.png",img1)
                # return ((1-score1)+(1-s))/2

            return 1-score1
        except Exception as e: # 嘿嘿
            return 0


    # 将物体摆放在相机坐标系的z轴上，旋转rx和ry，求得与参考图的相似度
    def get_values(self,z,rx,ry,alpha=1,show = False,only_one = False):
        try:
            PC1O = self.helper.compute_pose(z,rx,ry)
            img1 = self.render_silhouette(PC1O,id=0) # pose1的未矫正图像，也就是当前猜测视角下的图像，不考虑旋转
            R1,score1,s1,r1 = self.icp_rotation_PaI(img1,id=0,show=show,only_one = only_one)
            if only_one:
                # return 0.5,0,0
                return score1,s1,r1
            deltaP = np.identity(4)
            deltaP[:3,:3] = R1.T
            PC1O = deltaP@PC1O # 根据猜测的旋转，更新pose1

            PC2O = self.PC1O2PC2O(PC1O,self.PC1C2) # 根据pose1得出pose2
            img2 = self.render_silhouette(PC2O,id=1)
            R2,score2,s2,r2 = self.icp_rotation_PaI(img2,id=1,show=show,only_one = only_one)
            beta = 1-alpha
            return score1*alpha+score2*beta,s1*alpha+s2*beta,r1*alpha+r2*beta 
        except Exception as e: # 嘿嘿
            return 0,0,0
    
    # 将物体摆放在相机坐标系的z轴上，旋转rx和ry，求得与参考图的相似度
    def get_values_sec(self,z,rx,ry,alpha=1,show = False,only_one = False):
        try:
            PC2O = self.helper.compute_pose(z,rx,ry)
            img2 = self.render_silhouette(PC2O,id=1) # pose1的未矫正图像，也就是当前猜测视角下的图像，不考虑旋转
            R2,score2,s2,r2 = self.icp_rotation_PaI(img2,id=1,show=show,only_one = only_one)
            if only_one:
                return score2,s2,r2
            deltaP = np.identity(4)
            deltaP[:3,:3] = R2.T
            PC2O = deltaP@PC2O # 根据猜测的旋转，更新pose1

            PC1O = self.PC1C2@PC2O
            img1 = self.render_silhouette(PC1O,id=0)
            R1,score1,s1,r1 = self.icp_rotation_PaI(img1,id=0,show=show,only_one = only_one)
            beta = 1-alpha
            return score2*alpha+score1*beta,s2*alpha+s1*beta,r2*alpha+r1*beta 
        except Exception as e: # 嘿嘿
            return 0,0,0
    
    # 用虚拟相机1的zrxry计算俩相机坐标系中的物体位姿
    def get_pred_poses(self,z,rx,ry):
        PC1O = self.helper.compute_pose(z,rx,ry)
        img1 = self.render_silhouette(PC1O,id=0)
        #img = cv2.resize(img1,None,fx=0.25,fy=0.25)
        #cv2.imshow("123",img)
        #cv2.waitKey(0)
        # print("!!!!",z,rx,ry)
        R1,score1 = self.icp_rotation_PaI(img1,id=0)
        deltaP = np.identity(4)
        deltaP[:3,:3] = R1.T
        PC1O = deltaP@PC1O # 根据猜测的旋转，更新pose1
        return PC1O

    # 用虚拟相机2的zrxry计算俩相机坐标系中的物体位姿
    def get_pred_poses_sec(self,z,rx,ry):
        PC2O = self.helper.compute_pose(z,rx,ry)
        img2 = self.render_silhouette(PC2O,id=1) 
        R2,score2,s2,r2 = self.icp_rotation_PaI(img2,id=1)
        deltaP = np.identity(4)
        deltaP[:3,:3] = R2.T
        PC2O = deltaP@PC2O
        PC1O = self.PC1C2@PC2O
        return PC2O,PC1O

    # 默认是只求一个值
    def _search_fine_rxry(self,z,rx_init,ry_init,step,stop_step,func,only_one=True,alpha=1):
        rx = rx_init
        ry = ry_init
        last_move_step = (0,0)
        while True:
            if step<stop_step:
                break
            score,s,r = func(z,rx,ry,only_one=only_one)
            # print(rx,ry,score,s,r)
            score_l,s_l,r_l = func(z,rx-step,ry,only_one=only_one,alpha=alpha)
            score_r,s_r,r_r = func(z,rx+step,ry,only_one=only_one,alpha=alpha)
            score_u,s_u,r_u = func(z,rx,ry+step,only_one=only_one,alpha=alpha)
            score_d,s_d,r_d = func(z,rx,ry-step,only_one=only_one,alpha=alpha)
            # print(score_l,s_l,r_l)
            # print(score_r,s_r,r_r)
            # print(score_u,s_u,r_u)
            # print(score_d,s_d,r_d)
            lis = [[score_l,s_l,0], [score_r,s_r,1], [score_u,s_u,2], [score_d,s_d,3],]
            
            # 原则：整体只往算过的更大的地方走
            # 原则：从score增大的里面筛选s最大值，如果没有，那么选择s的最大值。这个值再去和当前值比较
            candidate = []
            for val in lis:
                if val[0] > score:
                    candidate.append(val)
                if len(candidate) > 1:
                    if candidate[0][0]>candidate[1][0]:
                        candidate.remove(candidate[1])
                    else:
                        candidate.remove(candidate[0])
            
            # 如果不是空的
            if len(candidate) > 0 and (last_move_step[1]!=step or (last_move_step[0]+candidate[0][2]!=1 and last_move_step[0]+candidate[0][2]!=5) ):
                candidate = candidate[0]
                # 这个候选和当前的值去比较(策略需要各种都遍历试试)
                ## 如果比当前大，移动
                if candidate[2] == 0: # switch不会写
                    rx-=step
                elif candidate[2] == 1:
                    rx+=step
                elif candidate[2] == 2:
                    ry+=step
                elif candidate[2] == 3:
                    ry-=step
                last_move_step = (candidate[2],step)
            ## 否则步长折半，不移动
            else:
                step/=2
        return rx,ry


    # coarse全局搜,或局部搜
    def search_PSO(self,z,rx_init,ry_init,iterations = 400,p_num = 50,coarse = True,th = 0.02):
        if coarse:
            rxrys = self.helper.getRxRy(self.helper.fibonacci_sphere(p_num))
            pso = PSO(self.get_value,3,-1,iterations,[z[0],rx_init-180,ry_init-180],[z[1],rx_init+180,ry_init+180],c1=0.5,c2=0.5,rxrys=rxrys,th=th)
        else:
            pso = PSO(self.get_value,3,p_num,iterations,[z[0],rx_init-5,ry_init-5],[z[1],rx_init+5,ry_init+5],c1=0.5,c2=0.5)
        pso.run()
        pso.update_gbest()
        pose = self.get_pred_poses(pso.gbest_x[0],pso.gbest_x[1],pso.gbest_x[2])
        return pose,(pso.gbest_x[0],pso.gbest_x[1],pso.gbest_x[2])

    def pose_es(self,znear,zfar,iterations,p_num,th):
        pose_,(z,rx,ry) = self.search_PSO((znear,zfar),0,0,iterations=iterations,p_num = p_num,coarse = True,th=th)
        print("pose_",pose_)
        qwe = self.renderer.deltaP
        P = np.eye(4)
        P[:3,3] = -qwe.T
        pose = pose_@P
        return pose,pose_
