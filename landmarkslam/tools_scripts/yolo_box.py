import cv2
import os
import glob

# ================= 1. 配置路径 =================
# 替换为你切片后存放单目图片的文件夹路径
image_dir = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data" 
output_txt_path = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/yolo_simulated_bboxes.txt"

# 获取所有图片并按名字排序
image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + 
                     glob.glob(os.path.join(image_dir, "*.png")))

if not image_paths:
    print(f"❌ 在 {image_dir} 中没有找到图片，请检查路径！")
    exit()

print(f"✅ 找到 {len(image_paths)} 张图片，开始连续标注...")
print("==================================================")
print("💡 操作说明：")
print("  - 鼠标左键拖拽：画框（记得把立柱也包进去一点）")
print("  - [空格] 或 [回车]：确认保存当前框，并自动进入下一张")
print("  - [c] 键：重画当前框（取消选择）")
print("  - [s] 键：跳过当前图（比如这张图里完全看不到标志牌）")
print("  - [q] 键：保存并提前退出标注")
print("==================================================")

# ================= 2. 开始交互式标注 =================
# 打开文件准备写入 (使用 'w' 模式会覆盖旧文件，如果想追加可以用 'a')
with open(output_txt_path, 'w', encoding='utf-8') as f:
    # 写入表头，方便后续使用 pandas 或 numpy 读取
    f.write("# filename, u_min(左上x), v_min(左上y), u_max(右下x), v_max(右下y)\n")
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        # 核心交互函数：cv2.selectROI
        # fromCenter=False: 从左上角拉到右下角
        # showCrosshair=True: 显示十字准星辅助对齐
        bbox = cv2.selectROI("Simulate YOLO BBox (Space: Save | S: Skip | Q: Quit)", 
                             img, fromCenter=False, showCrosshair=True)
        
        u_min, v_min, w, h = bbox
        
        # 键盘事件判断逻辑
        # selectROI 内部按了空格/回车确认后，会返回非零宽高。如果宽高为0，说明用户按了 'c' 或者 's'
        if w == 0 or h == 0:
            # 额外加一个键盘捕获，看看用户是不是想彻底退出
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("\n🛑 用户提前终止标注。")
                break
            elif key == ord('s'):
                print(f"⏭️ 跳过: {filename}")
                continue
            else:
                print(f"⏭️ 未画框，跳过: {filename}")
                continue
        
        # ================= 3. 计算 4 个角点并保存 =================
        u_max = u_min + w
        v_max = v_min + h
        
        # 写入 txt 文件 (格式: 文件名, x_min, y_min, x_max, y_max)
        f.write(f"{filename},{u_min},{v_min},{u_max},{v_max}\n")
        f.flush() # 实时刷入硬盘，防止中途崩溃白标了
        
        print(f"✅ 已保存 {filename} -> BBox: [{u_min}, {v_min}, {u_max}, {v_max}]")

cv2.destroyAllWindows()
print("==================================================")
print(f"🎉 标注全部完成！BBox 数据已安全保存在: {output_txt_path}")