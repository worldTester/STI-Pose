import renderer
import numpy as np
import cv2
import time
import pyrr

def GetR(rx,ry,rz)->np.ndarray:
    rx = np.array(pyrr.Matrix44.from_x_rotation(rx/180*3.14159))[:3,:3]
    ry = np.array(pyrr.Matrix44.from_y_rotation(ry/180*3.14159))[:3,:3]
    rz = np.array(pyrr.Matrix44.from_z_rotation(rz/180*3.14159))[:3,:3]
    tmp = rz.dot(ry).dot(rx)
    return tmp

#%% 

if __name__ == '__main__':
    ren_rgb = renderer.create_renderer(4096,3072,'vispy','rgb','phong') # 已经经过魔改，现在只出mask，这些光照模型是没有用上的
    ren_rgb.add_object(0,"../models/2.ply")

    #%%
    st = time.time()
    for i in range(1):
        # R = GetR(40,50,40)
        R = GetR(40,50,0)
        res = ren_rgb.render_object(0,R,np.array([0,0,300]),1957.289,1957.043,2048.868,1524.94)
    print(time.time()-st)
    img = res['rgb']
    cv2.imwrite("1.png",img)
    # cv2.imshow("hhh.png",img)
    # cv2.waitKey(0)

    # %%
