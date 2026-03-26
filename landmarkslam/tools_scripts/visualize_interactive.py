import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2

def interactive_visualize(txt_file, img_dir):
    if not os.path.exists(txt_file):
        print(f"Error: {txt_file} not found.")
        return

    print("Loading point cloud data...")
    frame_data = {}
    frame_order = []
    all_points = []
    
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 7: # X Y Z R G B Frame_name
                pt = [float(parts[0]), float(parts[1]), float(parts[2])]
                col = [float(parts[3])/255.0, float(parts[4])/255.0, float(parts[5])/255.0]
                frame_name = parts[6]
                
                if frame_name not in frame_data:
                    frame_data[frame_name] = {'pts': [], 'cols': []}
                    frame_order.append(frame_name)
                    
                frame_data[frame_name]['pts'].append(pt)
                frame_data[frame_name]['cols'].append(col)
                all_points.append(pt)
                
    all_points = np.array(all_points)
    if len(all_points) == 0:
        print("No valid points found.")
        return

    # === Pre-calculate Bounds based on all filtered points ===
    # 强制不使用 5% percentile 剪裁边界，原先的逻辑会把相机的远点当成异常点切除掉
    # 直接使用真实点的 min / max 来包含所有目标点(包括较远的相机轨迹点)
    filtered_points = all_points
    
    if len(filtered_points) == 0:
        filtered_points = all_points

    max_range = np.array([filtered_points[:, 0].max()-filtered_points[:, 0].min(), 
                          filtered_points[:, 1].max()-filtered_points[:, 1].min(), 
                          filtered_points[:, 2].max()-filtered_points[:, 2].min()]).max() / 2.0
    
    mid_x = (filtered_points[:, 0].max()+filtered_points[:, 0].min()) * 0.5
    mid_y = (filtered_points[:, 1].max()+filtered_points[:, 1].min()) * 0.5
    mid_z = (filtered_points[:, 2].max()+filtered_points[:, 2].min()) * 0.5

    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(mid_x - max_range*1.1, mid_x + max_range*1.1)
    ax.set_ylim(mid_y - max_range*1.1, mid_y + max_range*1.1)
    ax.set_zlim(mid_z - max_range*1.1, mid_z + max_range*1.1)
    ax.invert_yaxis()
    ax.invert_zaxis()
    ax.set_title("Interactive Point Cloud")
    
    cv2.namedWindow("Frame Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame Image", 1000, 750)
    
    print("\n" + "="*50)
    print(" === 交互式查看说明 ===")
    print(" 1. 请点击选中弹出的 'Frame Image' 图片窗口。")
    print(" 2. 按 空格键 (SPACE) 将一帧一帧地显示特征点。")
    print(" 3. 此时图表窗口会自动累加该帧生成的3D点，图片窗口也会切换为对应帧！")
    print(" 4. 按 'q' 键或 'ESC' 退出。")
    print("="*50 + "\n")

    cumulative_pts = 0

    for frame_name in frame_order:
        data = frame_data[frame_name]
        pts = np.array(data['pts'])
        cols = np.array(data['cols'])
        cumulative_pts += len(pts)
        
        # 1. Update 3D plot
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=cols, s=15.0, marker='o', alpha=0.8)
        ax.set_title(f"Accumulating... Frame: {frame_name} | Total PTS: {cumulative_pts}")
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # 2. Show 2D Image
        # 我们把追加了后缀的 _camera 等也还原回原来的真实图片名称
        real_img_name = frame_name.replace("_camera", "").replace("_corner", "")
        img_path = os.path.join(img_dir, real_img_name)
        img = cv2.imread(img_path)
        
        if img is not None:
            # 添加一些文字在左上角
            cv2.putText(img, f"Frame: {frame_name}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(img, f"Extracted Features: {len(pts)}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow("Frame Image", img)
        else:
            print(f"Warning: Image {img_path} not found.")
            
        # Wait for user input (Space or Q)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 32: # Space key
                break
            elif key == 27 or key == ord('q'): # ESC or q
                print("Exiting interactive viewer...")
                cv2.destroyAllWindows()
                plt.close('all')
                return
                
    print("Finished showing all frames.")
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="/home/zah/ORB_SLAM3-master/landmarkslam/output/slam_py_yolo/sign_cloud_py_sign.txt")
    parser.add_argument("--images", default="/home/zah/ORB_SLAM3-master/landmarkslam/data/20260321_111801/")
    args = parser.parse_args()
    interactive_visualize(args.file, args.images)
