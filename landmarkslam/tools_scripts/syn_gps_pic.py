import os
import glob
import csv

def parse_nmea_lat_lon(lat_str, lat_dir, lon_str, lon_dir):
    """
    将 NMEA 格式的经纬度 (DDMM.MMMMM) 转换为十进制角度
    """
    if not lat_str or not lon_str:
        return None, None
    try:
        lat_deg = float(lat_str[:2])
        lat_min = float(lat_str[2:])
        lat = lat_deg + (lat_min / 60.0)
        if lat_dir == 'S': lat = -lat
            
        lon_deg = float(lon_str[:3])
        lon_min = float(lon_str[3:])
        lon = lon_deg + (lon_min / 60.0)
        if lon_dir == 'W': lon = -lon
            
        return lat, lon
    except ValueError:
        return None, None

def load_gps_data(gps_file_path):
    """
    读取 GPS 文件，提取时间戳和有效的经纬度。
    """
    gps_records = []
    
    with open(gps_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'gps_time_ns:' not in line or 'gps_raw:' not in line:
                continue
            
            try:
                # 1. 分割提取时间和 NMEA 数据
                parts = line.strip().split(',gps_raw:')
                ts_ns = int(parts[0].replace('gps_time_ns:', '').strip())
                nmea_sentence = parts[1]
                nmea_parts = nmea_sentence.split(',')
                
                # 2. 解析 GNGGA 或 GPGGA (获取高精度位置)
                if nmea_parts[0] in ['$GNGGA', '$GPGGA']:
                    if len(nmea_parts) > 5 and nmea_parts[2] and nmea_parts[4]:
                        lat, lon = parse_nmea_lat_lon(nmea_parts[2], nmea_parts[3], nmea_parts[4], nmea_parts[5])
                        if lat is not None:
                            gps_records.append((ts_ns, (lat, lon)))
                            
                # 3. 解析 GNRMC 或 GPRMC (作为备用定位)
                elif nmea_parts[0] in ['$GNRMC', '$GPRMC']:
                    if len(nmea_parts) > 6 and nmea_parts[2] == 'A' and nmea_parts[3] and nmea_parts[5]:
                        lat, lon = parse_nmea_lat_lon(nmea_parts[3], nmea_parts[4], nmea_parts[5], nmea_parts[6])
                        if lat is not None:
                            gps_records.append((ts_ns, (lat, lon)))
            except Exception as e:
                # 忽略解析错误的异常行
                continue

    # 按照时间戳从小到大排序
    gps_records.sort(key=lambda x: x[0])
    return gps_records

def get_image_timestamps(image_folder):
    """
    读取文件夹中的图片名并提取时间戳。
    """
    image_records = []
    image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))
    
    for path in image_paths:
        filename = os.path.basename(path)
        name_without_ext = os.path.splitext(filename)[0]
        try:
            ts_str = name_without_ext.split('_')[-1]
            ts_ns = int(ts_str)
            image_records.append((ts_ns, filename))
        except ValueError:
            pass
            
    image_records.sort(key=lambda x: x[0])
    return image_records

def sync_image_and_gps(image_records, gps_records, max_time_diff_ns=200000000):
    """
    执行时间戳硬对齐 (就近匹配)
    """
    if not gps_records:
        return []

    synced_results = []
    
    for img_ts, img_name in image_records:
        # 寻找时间戳差异最小的 GPS 记录
        closest_gps = min(gps_records, key=lambda x: abs(x[0] - img_ts))
        gps_ts, (lat, lon) = closest_gps
        time_diff = abs(img_ts - gps_ts)
        
        if time_diff <= max_time_diff_ns:
            synced_results.append({
                'image_name': img_name,
                'image_ts_ns': img_ts,
                'gps_ts_ns': gps_ts,
                'latitude': lat,
                'longitude': lon,
                'time_diff_ms': time_diff / 1e6
            })
            
    return synced_results

def export_to_csv(results, output_csv="synced_data.csv"):
    """
    将对齐结果导出为 CSV 文件
    """
    if not results:
        print("没有可导出的对齐数据！")
        return

    # ================= 🌟 修复：自动创建目标文件夹 =================
    out_dir = os.path.dirname(output_csv)
    # 如果 out_dir 不为空，且文件夹不存在，则自动创建它 (连同父目录一起创建)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    # ==============================================================

    # 定义 CSV 表头
    headers = ['image_name', 'image_ts_ns', 'gps_ts_ns', 'latitude', 'longitude', 'time_diff_ms']

    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for row in results:
            writer.writerow({
                'image_name': row['image_name'],
                'image_ts_ns': row['image_ts_ns'],
                'gps_ts_ns': row['gps_ts_ns'],
                'latitude': f"{row['latitude']:.7f}",
                'longitude': f"{row['longitude']:.7f}",
                'time_diff_ms': f"{row['time_diff_ms']:.2f}"
            })
            
    print(f"\n✅ 成功！对齐结果已保存到文件: {os.path.abspath(output_csv)}")

if __name__ == "__main__":
    # ================= 1. 配置路径 =================
    GPS_LOG_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/gps_log.txt"      
    IMAGE_FOLDER = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243"            
    OUTPUT_CSV = "/home/zah/ORB_SLAM3-master/landmarkslam/output/gsp_pic_synced/synced_data.csv"     # 导出的 CSV 文件名
    
    # 最大允许的时间差 (单位：纳秒)，200000000纳秒 = 0.2秒
    MAX_DIFF_NS = 200000000 
    # ===============================================
    
    if not os.path.exists(GPS_LOG_FILE):
         print(f"❌ 找不到 GPS 日志文件: {GPS_LOG_FILE}")
    elif not os.path.exists(IMAGE_FOLDER):
        print(f"❌ 找不到图片文件夹: {IMAGE_FOLDER}")
    else:
        print("⏳ 正在读取数据并提取时间戳...")
        gps_data = load_gps_data(GPS_LOG_FILE)
        img_data = get_image_timestamps(IMAGE_FOLDER)
        
        print(f"📊 提取到 {len(img_data)} 张图片 和 {len(gps_data)} 条有效 GPS 记录。")
        
        print("🔄 正在执行时间戳对齐...")
        results = sync_image_and_gps(img_data, gps_data, MAX_DIFF_NS)
        
        print(f"✔️ 成功对齐 {len(results)}/{len(img_data)} 张图片。")
        
        # 导出为 CSV
        export_to_csv(results, OUTPUT_CSV)