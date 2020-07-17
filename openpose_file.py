import os
cmd = "/content/openpose/build/examples/openpose/openpose.bin  --image_dir "/content/drive/My Drive/CIHP_PGN-master/dataset/images/" --display 0 --render_pose 0 --model_pose COCO --write_json "/content/drive/My Drive/CIHP_PGN-master/dataset/pose_coco/" "
os.system(cmd)