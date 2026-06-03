#!/home/zah/ORB_SLAM3-master/landmarkslam/yolo_venv/bin/python3
"""
路牌四角点标注工具 — 鼠标点 4 个角点，自动算中心

用法:
  python3 annotate_corners.py <图片目录> [标注文件路径]

格式:
  文件名 x1,y1 x2,y2 x3,y3 x4,y4
  003613.png 257,258 390,260 382,308 249,305
"""
import os, sys, cv2

def load_annotations(anno_path):
    annos = {}
    if os.path.exists(anno_path):
        with open(anno_path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) == 5:
                    fname = parts[0]
                    pts = [tuple(map(int, p.split(','))) for p in parts[1:5]]
                    annos[fname] = pts
    return annos

def save_annotations(anno_path, annos):
    with open(anno_path, 'w') as f:
        for fname, pts in annos.items():
            f.write(f"{fname} {pts[0][0]},{pts[0][1]} {pts[1][0]},{pts[1][1]} "
                    f"{pts[2][0]},{pts[2][1]} {pts[3][0]},{pts[3][1]}\n")

def main():
    if len(sys.argv) < 2:
        print("用法: python3 annotate_corners.py <image_dir> [anno_file]")
        sys.exit(1)
    
    img_dir = sys.argv[1]
    anno_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(img_dir, "corners.txt")
    
    images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.png')])
    if not images:
        print(f"❌ 无图片: {img_dir}")
        sys.exit(1)
    
    annos = load_annotations(anno_path)
    total = len(images)
    print(f"📂 {total} 张 | 已标注: {len(annos)} | {anno_path}")
    
    idx, current_pts = 0, []
    
    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(current_pts) < 4:
            current_pts.append((x, y))
    
    win = "Corners | SPACE=start | A/D=prev/next | Q=save"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb)
    
    while True:
        img = cv2.imread(os.path.join(img_dir, images[idx]))
        if img is None: continue
        
        disp, fname = img.copy(), images[idx]
        
        # 已有标注
        if fname in annos:
            pts = annos[fname]
            for i, (x, y) in enumerate(pts):
                cv2.circle(disp, (x, y), 6, (0, 255, 255), -1)
                cv2.putText(disp, str(i+1), (x+8, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            for i in range(4):
                cv2.line(disp, pts[i], pts[(i+1)%4], (0, 255, 255), 2)
            cx = sum(p[0] for p in pts) // 4
            cy = sum(p[1] for p in pts) // 4
            cv2.line(disp, pts[0], pts[2], (0, 255, 0), 1)
            cv2.line(disp, pts[1], pts[3], (0, 255, 0), 1)
            cv2.drawMarker(disp, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        
        # 正在标注
        for i, (x, y) in enumerate(current_pts):
            cv2.circle(disp, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(disp, str(i+1), (x+8, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            if i > 0: cv2.line(disp, current_pts[i-1], current_pts[i], (255, 0, 0), 1)
            if i == 3: cv2.line(disp, current_pts[3], current_pts[0], (255, 0, 0), 1)
        
        status = f"Clicking {len(current_pts)+1}/4" if current_pts else f"{idx}/{total-1} {fname}"
        if not current_pts: status += " ✅" if fname in annos else " ❌"
        cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(disp, "[A]prev [D]next [SPACE]start [Q]save",
                    (10, disp.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(win, disp)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q'):
            if current_pts: print("⚠️ 放弃未完成标注"); current_pts = []
            break
        elif key == ord('d') or key == 83:
            if current_pts: print("⚠️ 放弃"); current_pts = []
            idx = (idx + 1) % total
        elif key == ord('a') or key == 81:
            if current_pts: print("⚠️ 放弃"); current_pts = []
            idx = (idx - 1) % total
        elif key == 32:
            if len(current_pts) == 4:
                annos[fname] = current_pts[:]
                print(f"  ✅ {fname}: {current_pts}"[:80])
                current_pts = []
            elif len(current_pts) > 0:
                print(f"  ⚠️ 还需要 {4-len(current_pts)} 个点")
            else:
                current_pts = []
                annos.pop(fname, None)
                print(f"  🎯 {fname}: 依次点 4 角点(顺时针)")
        elif key == 27 and current_pts:
            current_pts = []; print(f"  ⚠️ 取消")
    
    cv2.destroyAllWindows()
    save_annotations(anno_path, annos)
    print(f"\n✅ 保存 {len(annos)} 个标注 → {anno_path}")

if __name__ == "__main__":
    main()
