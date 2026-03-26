import sys

filepath = '/home/zah/ORB_SLAM3-master/landmarkslam/run_landmarkslam_yolo.cc'
with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('printAndLog("\\n[Landmark', 'printAndLog("\\\\n[Landmark')
text = text.replace('printAndLog("\\n[Landmark Extractor]', 'printAndLog("\\\\n[Landmark Extractor]')
text = text.replace('printAndLog("\\n', 'printAndLog("\\\\n')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)
