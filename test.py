from SilhouettePE import *
import cv2
import numpy as np

if __name__ == '__main__':
    img_size = (640,480) # image_size
    K = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]]) # camera intrinsic matrix K
    p = Process(img_size,K,1) # the STI-Pose algorithm
    p.set_model("model.ply") # the 3D model

    # ※※※adjust the pose_gt for more experiments.※※※
    pose_gt = p.helper.compute_pose(400,90,90) # a pose that we think it the ground truth

    ref_silhouette = p.render_silhouette(pose_gt) # the reference silhouette

    cv2.imwrite("ref_silhouette.png",ref_silhouette)

    p.set_ref(ref_silhouette)
    pose_pred,_ = p.pose_es(50,1000,100,20,0.02) # z_near, z_far, iterations, number of particles, the threshold controlling the termination of STI-Pose
    result_rendered = p.render_silhouette(_)
    print("The ground truth pose is:")
    print(pose_gt)
    print("The predicted object pose is:")
    print(pose_pred)
    cv2.imwrite("result_silhouette.png",result_rendered)
