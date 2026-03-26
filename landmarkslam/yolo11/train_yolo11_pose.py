from ultralytics import YOLO

def main():
    # 加载 YOLO11 Nano Pose 模型 (专用于关键点检测)
    # 如果没有预训练权重，它会自动下载
    model = YOLO('yolo11n-pose.pt') 

    # 开始训练
    results = model.train(
        data=r'd:\zah\Latex\yolo_dataset\data.yaml', 
        epochs=50,             
        imgsz=640,              
        batch=16,               
        device=0,               # 如果是用 CPU 训练，请改为 device='cpu'
        workers=2,              # Windows下建议设为 0-2
        project='road_sign_pose_project', 
        name='yolo11n_pose_run1',
        save_period=10,         # 每 10 轮保存一次权重
        # 关键点配置 (虽然模型会自动适配，但显式指定是个好习惯)
        # kpt_shape: [关键点数量, 维度] -> [4, 3] (x, y, visibility)
        # 但在 ultralytics 中，通常由 data.yaml 中的 kpt_shape 定义，或者自动推断
    )

if __name__ == '__main__':
    main()
