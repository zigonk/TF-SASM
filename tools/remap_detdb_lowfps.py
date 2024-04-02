import json

det_db = json.load(open("data/Dataset/mot/det_db_motrv2.json"))

def convert_to_lowfps_key(key, skip=5):
    parts = key.split("/")
    print(parts)
    parts[0] = "DanceTrack_lowfps_skip5"
    # 0001/img1/000001.txt
    parts[-1] = f"{int(parts[-1].split('.')[0]) // skip + 1:08d}.txt"
    return "/".join(parts)

det_db_lowfps = {}
for key, value in sorted(det_db.items()):
    if key.startswith("DanceTrack"):    
        if convert_to_lowfps_key(key) not in det_db_lowfps:
            print(key, convert_to_lowfps_key(key))
            det_db_lowfps[convert_to_lowfps_key(key)] = value

with open("data/Dataset/mot/det_db_motrv2_lowfps_skip5.json", 'w') as f:
    json.dump(det_db_lowfps, f)


