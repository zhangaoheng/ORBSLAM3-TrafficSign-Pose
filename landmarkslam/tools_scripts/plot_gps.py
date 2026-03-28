import folium
import os

def parse_nmea_lat_lon(lat_str, lat_dir, lon_str, lon_dir):
    """
    将 NMEA 格式的经纬度 (DDMM.MMMMM) 转换为十进制角度
    """
    if not lat_str or not lon_str:
        return None, None
        
    # 纬度: 前两位是度，后面是分
    lat_deg = float(lat_str[:2])
    lat_min = float(lat_str[2:])
    lat = lat_deg + (lat_min / 60.0)
    if lat_dir == 'S':
        lat = -lat
        
    # 经度: 前三位是度，后面是分
    lon_deg = float(lon_str[:3])
    lon_min = float(lon_str[3:])
    lon = lon_deg + (lon_min / 60.0)
    if lon_dir == 'W':
        lon = -lon
        
    return lat, lon

def extract_path_from_log(file_path):
    coordinates = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'gps_raw:' not in line:
                continue
                
            # 提取 gps_raw 后面的 NMEA 语句
            nmea_sentence = line.strip().split('gps_raw:')[1]
            parts = nmea_sentence.split(',')
            
            # 使用 $GNGGA 或 $GPGGA 语句来获取位置信息
            if parts[0] in ['$GNGGA', '$GPGGA']:
                lat_str = parts[2]
                lat_dir = parts[3]
                lon_str = parts[4]
                lon_dir = parts[5]
                
                lat, lon = parse_nmea_lat_lon(lat_str, lat_dir, lon_str, lon_dir)
                if lat is not None and lon is not None:
                    coordinates.append((lat, lon))
                    
    return coordinates

def create_map(coordinates, output_file="gps_path.html"):
    if not coordinates:
        print("未在文件中找到有效的 GPS 坐标点！")
        return

    print(f"成功提取 {len(coordinates)} 个坐标点。")
    
    # 以第一个点为中心生成地图
    start_point = coordinates[0]
    m = folium.Map(location=start_point, zoom_start=18)
    
    # 标记起点和终点
    folium.Marker(coordinates[0], popup="起点", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(coordinates[-1], popup="终点", icon=folium.Icon(color="red")).add_to(m)
    
    # 绘制轨迹连线
    folium.PolyLine(
        locations=coordinates,
        color='blue',
        weight=5,
        opacity=0.8
    ).add_to(m)
    
    # 保存为 HTML
    m.save(output_file)
    print(f"地图已生成！请在浏览器中打开: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    # 假设你的数据存放在 gps_data.txt 中
    log_file = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/gps_log.txt" 
    
    if not os.path.exists(log_file):
        print(f"请先创建 {log_file} 文件，并粘贴你的 GPS 数据。")
    else:
        coords = extract_path_from_log(log_file)
        create_map(coords)