import cv2
import os

# 创建存放图片的文件夹
folder = "calib_images"
if not os.path.exists(folder):
    os.makedirs(folder)

# 打开摄像头 (0为默认摄像头，如果是外接摄像头可能是1或2)
# 如果你需要特定分辨率，可以在这里设置，例如：
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap = cv2.VideoCapture(0)

count = 0
print("====================================")
print("📸 摄像头已开启！")
print("👉 请将标定板放在镜头前。")
print("👉 按下【空格键】抓拍保存图片。")
print("👉 拍满 15-20 张后，按【q】键退出。")
print("====================================")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取画面，请检查摄像头连接！")
        break

    cv2.imshow('Camera Calibration Capture', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # 按空格键保存
        img_name = os.path.join(folder, f"calib_{count:02d}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"✅ 已保存: {img_name} (共 {count+1} 张)")
        count += 1
    elif key == ord('q'):  # 按 q 键退出
        break

cap.release()
cv2.destroyAllWindows()