import json
DET_DB_DIR = "data/Dataset/mot/det_db"
det_db = json.load(open(f"{DET_DB_DIR}/det_db_motrv2.json"))

def convert_to_stack_frame(key, skip=5):
    parts = key.split("/")
    parts[0] = "DanceTrack_variants/stack_10"
    return "/".join(parts)

det_db_stack_frame = {}
for key, value in sorted(det_db.items()):
    if key.startswith("DanceTrack"):    
        if convert_to_stack_frame(key) not in det_db_stack_frame:
            print(key, convert_to_stack_frame(key))
            det_db_stack_frame[convert_to_stack_frame(key)] = value

with open(f"{DET_DB_DIR}/det_db_motrv2_stack_10.json", 'w') as f:
    json.dump(det_db_stack_frame, f)


