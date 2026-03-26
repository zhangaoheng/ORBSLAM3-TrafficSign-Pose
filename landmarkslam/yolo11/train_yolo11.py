from ultralytics import YOLO

def main():
    # 加载 YOLO11 Nano 模型 (最快)
    model = YOLO('yolo11n.pt') 

    # 开始训练
    # 注意: data 参数需要指向您的 data.yaml 文件的绝对路径或相对路径
    results = model.train(
        data=r'd:\zah\Latex\yolo_dataset\data.yaml', 
        epochs=100,             
        imgsz=640,              
        batch=16,               
        device=0,               # 如果是用 CPU 训练，请改为 device='cpu'
        workers=2,              # Windows下如果报错，可以将 workers 改为 0 或 1
        project='road_sign_project', 
        name='yolo11n_run1'     
    )

if __name__ == '__main__':
    main()