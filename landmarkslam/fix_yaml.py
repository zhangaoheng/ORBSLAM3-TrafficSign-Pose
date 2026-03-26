with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "r") as f:
    code = f.read()

code = code.replace("fx = 458.654", "fx = 935.307")
code = code.replace("fy = 457.296", "fy = 935.307")
code = code.replace("cx = 367.215", "cx = 960.0")
code = code.replace("cy = 248.375", "cy = 540.0")

# Update ORB parameters based on YAML
code = code.replace("nfeatures=3000", "nfeatures=1000")
code = code.replace("edgeThreshold=31", "edgeThreshold=20")
code = code.replace("fastThreshold=20", "fastThreshold=7")
code = code.replace("patchSize=31", "patchSize=20") 

with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "w") as f:
    f.write(code)
