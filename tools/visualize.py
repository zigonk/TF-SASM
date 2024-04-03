# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


from collections import defaultdict
from glob import glob
import json
import os
import cv2
import subprocess
from tqdm import tqdm


def get_color(i):
    return [(i * 23 * j + 43) % 255 for j in range(3)]


# with open("./data/Dataset/mot/det_db_oc_sort_full.json") as f:
#     det_db = json.load(f)

gt_dir = "data/Dataset/mot/DanceTrack_variants/stack_10/val"


def process(trk_path, img_list, output="output.mp4"):
    h, w, _ = cv2.imread(img_list[0]).shape
    command = [
        "/usr/bin/ffmpeg",
        '-y',  # overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',  # size of one frame
        '-pix_fmt', 'bgr24',
        '-r', '20',  # frames per second
        '-i', '-',  # The imput comes from a pipe
        '-s', f'{w//2*2}x{h//2*2}',
        '-an',  # Tells FFMPEG not to expect any audio
        '-loglevel', 'error',
        '-crf', '26',
        '-pix_fmt', 'yuv420p'
    ]
    writing_process = subprocess.Popen(
        command + [output], stdin=subprocess.PIPE)

    tracklets = defaultdict(list)
    gt_tracklet = defaultdict(list)
    for line in open(trk_path):
        t, id, *xywhs = line.split(',')[:7]
        t, id = map(int, (t, id))
        x, y, w, h, s = map(float, xywhs)
        tracklets[t].append((id, *map(int, (x, y, x+w, y+h))))

    gt_path = os.path.join(gt_dir, trk_path.split(
        '/')[-1]).replace('txt', 'gt/gt.txt')
    if os.path.exists(gt_path):
        for line in open(gt_path):
            t, id, *xywhs = line.split(',')[:7]
            t, id = map(int, (t, id))
            x, y, w, h, s = map(float, xywhs)
            gt_tracklet[t].append((id, *map(int, (x, y, x+w, y+h))))

    for i, path in enumerate(tqdm(sorted(img_list))):
        im = cv2.imread(path)
        # for det in det_db[path.replace('.jpg', '.txt')]:
        #     x1, y1, w, h, _ = map(int, map(float, det.strip().split(',')))
        #     im = cv2.rectangle(im, (x1, y1), (x1+w, y1+h), (255, 255, 255), 6)
        # for j, x1, y1, x2, y2 in gt_tracklet[i + 1]:
        #     im = cv2.rectangle(im, (x1, y1), (x2, y2), (255, 255, 255), 4)
        #     im = cv2.putText(im, f"{j}", (x1 + 10, y1 + 30),
        #                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        for j, x1, y1, x2, y2 in tracklets[i + 1]:
            im = cv2.rectangle(im, (x1, y1), (x2, y2), get_color(j), 4)
            im = cv2.putText(im, f"{j}", (x1 + 10, y1 + 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, get_color(j), 2)
        writing_process.stdin.write(im.tobytes())


if __name__ == '__main__':
    METHOD="sam_occlusion_discard-test"
    track_dir = "./outputs/tracker_" + METHOD + "/"
    method_name = "motrv2_" + METHOD
    DATASET_NAME = "DanceTrack"
    DATA_SPLIT = "test"
    os.makedirs(f'visualize/{method_name}', exist_ok=True)
    jobs = os.listdir(track_dir)
    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    jobs = sorted(jobs)[rank::ws]
    for seq in jobs:
        print(seq)
        if (seq != "dancetrack0064.txt"):
            continue
        trk_path = track_dir + seq
        # trk_path = "/data/Dataset/mot/DanceTrack/val/dancetrack0010/gt/gt.txt"
        vid_path = f"data/Dataset/mot/{DATASET_NAME}/{DATA_SPLIT}/{seq[:-4]}/img1"
        if not os.path.exists(vid_path):
            continue
        img_list = glob(
            f"data/Dataset/mot/{DATASET_NAME}/{DATA_SPLIT}/{seq[:-4]}/img1/*.jpg")
        process(trk_path, img_list, f'visualize/{method_name}/{seq[:-4]}.mp4')
