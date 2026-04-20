import cv2
import os
import glob
import numpy as np

# 从你的核心模块中导入函数
from tools.mid import extract_four_lines_from_real_image, calculate_rectangle_center

def load_saved_rois(txt_path):
    """从 txt 文件中加载已经保存的 ROI"""
    rois = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                if line.strip():
                    # 解析逗号分隔的 x,y,w,h
                    x, y, w, h = map(int, line.strip().split(','))
                    rois.append((x, y, w, h))
        print(f">>> 📂 找到历史标注文件！已成功加载 {len(rois)} 个保存的框。")
    else:
        print(">>> 📝 未找到历史标注文件，将进入【手动标注模式】。")
    return rois

def process_sequence_with_cached_rois(folder_path):
    print(f">>> 正在读取文件夹: {folder_path}")
    
    # 1. 获取图片列表并排序
    images = sorted(glob.glob(os.path.join(folder_path, "*.png")) +
                    glob.glob(os.path.join(folder_path, "*.jpg")))

    if not images:
        print(f"❌ 找不到图片，请检查文件夹路径: {folder_path}")
        return

    print(f"✅ 共找到 {len(images)} 张图片。")

    # 2. 定义保存框坐标的 txt 文件路径 (保存在图片同一个目录下)
    roi_txt_path = os.path.join(folder_path, "saved_rois.txt")
    
    # 3. 加载已经保存的坐标
    saved_rois = load_saved_rois(roi_txt_path)

    # 4. 以追加模式打开 txt 文件，准备写入新画的框
    # 如果程序中途退出，已经画好的框也会被安全保存
    roi_file = open(roi_txt_path, 'a')

    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None: continue

        display_img = img.copy()
        current_roi = None

        # ==========================================
        # 核心逻辑：判断当前帧是否有保存的框
        # ==========================================
        if idx < len(saved_rois):
            # 已经保存过，直接读取！
            current_roi = saved_rois[idx]
            mode_text = "Auto Loaded"
            color_theme = (255, 255, 255) # 白色框表示读取的
        else:
            # 没保存过，暂停，弹窗要求手动框选！
            print(f">>> 👉 请手动框选第 {idx+1} 帧，按 [SPACE] 或 [ENTER] 确认，按 'c' 取消。")
            current_roi = cv2.selectROI(f"Manual Annotation - Frame {idx+1}", display_img, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(f"Manual Annotation - Frame {idx+1}")
            
            if current_roi == (0, 0, 0, 0):
                print("❌ 用户取消框选，结束当前运行。")
                break
            
            # 将新画的框立即保存到 txt 文件中
            roi_file.write(f"{current_roi[0]},{current_roi[1]},{current_roi[2]},{current_roi[3]}\n")
            roi_file.flush() # 强制写入硬盘
            
            mode_text = "Manual Drawn"
            color_theme = (0, 255, 255) # 黄色框表示刚刚手画的

        # 获取当前框的参数
        x, y, w, h = current_roi

        # ==========================================
        # 几何推演与中心点计算
        # ==========================================
        lines = extract_four_lines_from_real_image(img, current_roi)

        if lines is not None:
            line_top, line_bottom, line_left, line_right = lines
            center, corners = calculate_rectangle_center(line_top, line_bottom, line_left, line_right)

            if center is not None:
                cx, cy = center
                tl, tr, bl, br = corners

                # [画图] ROI 框
                cv2.rectangle(display_img, (x, y), (x+w, y+h), color_theme, 1)

                # [画图] 四条延长线 (蓝色)
                cv2.line(display_img, (line_top[0], line_top[1]), (line_top[2], line_top[3]), (255, 0, 0), 1)
                cv2.line(display_img, (line_bottom[0], line_bottom[1]), (line_bottom[2], line_bottom[3]), (255, 0, 0), 1)
                cv2.line(display_img, (line_left[0], line_left[1]), (line_left[2], line_left[3]), (255, 0, 0), 1)
                cv2.line(display_img, (line_right[0], line_right[1]), (line_right[2], line_right[3]), (255, 0, 0), 1)

                # [画图] 四个角点与对角线 (黄点 + 绿线)
                for pt in corners:
                    cv2.circle(display_img, pt, 5, (0, 255, 255), -1)
                cv2.line(display_img, tl, br, (0, 255, 0), 1)
                cv2.line(display_img, tr, bl, (0, 255, 0), 1)

                # [画图] 中心十字准星 (红色)
                cv2.line(display_img, (cx - 20, cy), (cx + 20, cy), (0, 0, 255), 2)
                cv2.line(display_img, (cx, cy - 20), (cx, cy + 20), (0, 0, 255), 2)
                cv2.circle(display_img, center, 4, (0, 0, 255), -1)

                cv2.putText(display_img, f"Frame {idx+1} | {mode_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_theme, 2)
            else:
                cv2.putText(display_img, f"Lost Center: Frame {idx+1}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.putText(display_img, f"Lost Lines: Frame {idx+1}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ==========================================
        # 播放控制
        # ==========================================
        cv2.imshow("Geometry Pipeline", display_img)

        # 如果是读历史数据，播放快一点 (30ms)；如果是手动画完，稍作停顿 (100ms) 让看看效果
        delay = 30 if idx < len(saved_rois) else 100
        
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            print(">>> 用户手动中止播放。")
            break

    roi_file.close()
    cv2.destroyAllWindows()
    print(">>> 序列处理完毕！")
    print(f">>> 所有的框已经安全保存在: {roi_txt_path}")

if __name__ == "__main__":
    # ==========================================
    # 改成你的图片序列文件夹路径
    # ==========================================
    SEQUENCE_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/mid"
    
    process_sequence_with_cached_rois(SEQUENCE_DIR)