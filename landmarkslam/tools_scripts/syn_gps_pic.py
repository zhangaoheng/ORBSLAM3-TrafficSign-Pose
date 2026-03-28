import os
import glob
import csv
import bisect

def parse_nmea_lat_lon(lat_str, lat_dir, lon_str, lon_dir):
    """将 NMEA 格式转换为十进制角度"""
    if not lat_str or not lon_str: return None, None
    try:
        lat = float(lat_str[:2]) + (float(lat_str[2:]) / 60.0)
        if lat_dir == 'S': lat = -lat
        lon = float(lon_str[:3]) + (float(lon_str[3:]) / 60.0)
        if lon_dir == 'W': lon = -lon
        return lat, lon
    except ValueError:
        return None, None

def load_gps_data(gps_file_path):
    """读取 GPS 文件，提取时间、经纬度、高度、定位质量"""
    gps_records = []
    
    with open(gps_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'gps_time_ns:' not in line or 'gps_raw:' not in line:
                continue
            
            try:
                parts = line.strip().split(',gps_raw:')
                ts_ns = int(parts[0].replace('gps_time_ns:', '').strip())
                nmea_parts = parts[1].split(',')
                
                # 1. 解析 GNGGA/GPGGA (包含经纬度和海拔高度)
                if nmea_parts[0] in ['$GNGGA', '$GPGGA']:
                    if len(nmea_parts) > 9 and nmea_parts[2] and nmea_parts[4]:
                        lat, lon = parse_nmea_lat_lon(nmea_parts[2], nmea_parts[3], nmea_parts[4], nmea_parts[5])
                        
                        # 🌟 提取海拔高度 (第9位) 和 定位质量 (第6位)
                        alt = float(nmea_parts[9]) if nmea_parts[9] else 0.0
                        quality = int(nmea_parts[6]) if nmea_parts[6] else 0
                        
                        if lat is not None:
                            gps_records.append((ts_ns, lat, lon, alt, quality))
                            
                # 2. 解析 GNRMC/GPRMC (作为备用，没有高度信息)
                elif nmea_parts[0] in ['$GNRMC', '$GPRMC']:
                    if len(nmea_parts) > 6 and nmea_parts[2] == 'A' and nmea_parts[3] and nmea_parts[5]:
                        lat, lon = parse_nmea_lat_lon(nmea_parts[3], nmea_parts[4], nmea_parts[5], nmea_parts[6])
                        if lat is not None:
                            # RMC 没有高度，默认填 0.0
                            gps_records.append((ts_ns, lat, lon, 0.0, 1))
            except Exception:
                continue

    # 按时间戳排序，这是二分查找的前提
    gps_records.sort(key=lambda x: x[0])
    return gps_records

def get_image_timestamps(image_folder):
    """读取图片名并提取时间戳"""
    image_records = []
    image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))
    for path in image_paths:
        filename = os.path.basename(path)
        try:
            ts_ns = int(os.path.splitext(filename)[0].split('_')[-1])
            image_records.append((ts_ns, filename))
        except ValueError:
            pass
    image_records.sort(key=lambda x: x[0])
    return image_records

def sync_image_and_gps(image_records, gps_records, max_time_diff_ns=200000000):
    """执行时间戳硬对齐"""
    if not gps_records: return []

    synced_results = []
    gps_times = [r[0] for r in gps_records]
    
    for img_ts, img_name in image_records:
        # 使用二分查找迅速定位到最近的 GPS 点
        idx = bisect.bisect_left(gps_times, img_ts)
        
        if idx == 0:
            closest_gps = gps_records[0]
        elif idx == len(gps_records):
            closest_gps = gps_records[-1]
        else:
            before = gps_records[idx - 1]
            after = gps_records[idx]
            closest_gps = before if (img_ts - before[0]) <= (after[0] - img_ts) else after
            
        # 🌟 解包出包含高度 (alt) 的数据
        gps_ts, lat, lon, alt, quality = closest_gps
        time_diff = abs(img_ts - gps_ts)
        
        if time_diff <= max_time_diff_ns:
            synced_results.append({
                'image_name': img_name,
                'image_ts_ns': img_ts,
                'gps_ts_ns': gps_ts,
                'latitude': lat,
                'longitude': lon,
                'altitude': alt,        # 🌟 将高度加入结果字典
                'fix_quality': quality, 
                'time_diff_ms': time_diff / 1e6
            })
            
    return synced_results

def export_to_csv(results, output_csv="synced_data.csv"):
    if not results: return
    
    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.exists(out_dir): os.makedirs(out_dir, exist_ok=True)

    # 🌟 CSV 表头加入 altitude 和 fix_quality
    headers = ['image_name', 'image_ts_ns', 'gps_ts_ns', 'latitude', 'longitude', 'altitude', 'fix_quality', 'time_diff_ms']

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
                'altitude': f"{row['altitude']:.3f}", # 🌟 写入高度，保留3位小数(毫米级)
                'fix_quality': row['fix_quality'],
                'time_diff_ms': f"{row['time_diff_ms']:.2f}"
            })
    print(f"\n✅ 成功！对齐结果已保存到文件: {os.path.abspath(output_csv)}")

if __name__ == "__main__":
    GPS_LOG_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/gps_log.txt"      
    IMAGE_FOLDER = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243"            
    OUTPUT_CSV = "/home/zah/ORB_SLAM3-master/landmarkslam/output/gsp_pic_synced/synced_data.csv"     
    MAX_DIFF_NS = 200000000 
    
    if not os.path.exists(GPS_LOG_FILE) or not os.path.exists(IMAGE_FOLDER):
        print("❌ 路径错误，请检查 GPS 文件或图片文件夹是否存在。")
    else:
        gps_data = load_gps_data(GPS_LOG_FILE)
        img_data = get_image_timestamps(IMAGE_FOLDER)
        print(f"📊 提取到 {len(img_data)} 张图片 和 {len(gps_data)} 条有效 GPS 记录。")
        results = sync_image_and_gps(img_data, gps_data, MAX_DIFF_NS)
        print(f"✔️ 成功对齐 {len(results)}/{len(img_data)} 张图片。")
        export_to_csv(results, OUTPUT_CSV)