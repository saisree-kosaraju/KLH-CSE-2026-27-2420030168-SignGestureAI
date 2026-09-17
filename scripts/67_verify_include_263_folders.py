from pathlib import Path
import re

ROOT = Path.cwd()
DATASET = ROOT / "datasets" / "INCLUDE"
VIDEO_EXTS = {".mp4",".avi",".mov",".mkv",".webm",".mpeg",".mpg",".m4v"}
OFFICIAL = ['actor', 'adult', 'afternoon', 'alive', 'alright', 'animal', 'artist', 'attack', 'author', 'baby', 'bad', 'bag', 'ball', 'bank', 'bathroom', 'beautiful', 'bed', 'bedroom', 'bicycle', 'biglarge', 'bill', 'bird', 'black', 'blind', 'blue', 'boat', 'book', 'box', 'boy', 'brother', 'brown', 'bus', 'camera', 'car', 'card', 'cat', 'cellphone', 'chair', 'cheap', 'child', 'city', 'clean', 'clock', 'clothing', 'cold', 'colour', 'computer', 'cool', 'court', 'cow', 'crowd', 'curved', 'daughter', 'dead', 'deaf', 'death', 'deep', 'dirty', 'doctor', 'dog', 'door', 'dream', 'dress', 'dry', 'election', 'energy', 'evening', 'exercise', 'exmonsoon', 'expensive', 'fall', 'family', 'famous', 'fan', 'fast', 'father', 'female', 'fish', 'flat', 'friday', 'friend', 'gift', 'girl', 'god', 'good', 'goodafternoon', 'goodevening', 'goodmorning', 'goodnight', 'grandfather', 'grandmother', 'green', 'grey', 'ground', 'gun', 'happy', 'hard', 'hat', 'he', 'healthy', 'heavy', 'hello', 'high', 'horse', 'hospital', 'hot', 'hour', 'house', 'howareyou', 'husband', 'i', 'india', 'it', 'job', 'key', 'king', 'kitchen', 'lamp', 'laptop', 'lawyer', 'letter', 'library', 'light', 'location', 'lock', 'long', 'loose', 'loud', 'low', 'male', 'man', 'manager', 'market', 'marriage', 'mean', 'medicine', 'minute', 'monday', 'money', 'month', 'morning', 'mother', 'mouse', 'narrow', 'neighbour', 'new', 'newspaper', 'nice', 'night', 'office', 'old', 'orange', 'page', 'paint', 'pant', 'paper', 'parent', 'park', 'patient', 'peace', 'pen', 'pencil', 'photograph', 'pink', 'plane', 'player', 'pleased', 'pocket', 'police', 'poor', 'president', 'price', 'priest', 'queen', 'quiet', 'raceethnicity', 'radio', 'red', 'religion', 'reporter', 'restaurant', 'rich', 'ring', 'sad', 'saturday', 'school', 'science', 'screen', 'season', 'second', 'secondnumber', 'secretary', 'shallow', 'she', 'shirt', 'shoes', 'short', 'sick', 'sign', 'sister', 'skirt', 'slow', 'smalllittle', 'soap', 'soft', 'soldier', 'son', 'sport', 'spring', 'storeorshop', 'streetorroad', 'strong', 'student', 'suit', 'summer', 'sunday', 'table', 'tall', 'teacher', 'team', 'technology', 'telephone', 'television', 'temple', 'thankyou', 'they', 'thick', 'thin', 'thursday', 'tight', 'time', 'today', 'tomorrow', 'tool', 'train', 'trainstation', 'trainticket', 'transportation', 'truck', 'tshirt', 'tuesday', 'ugly', 'university', 'waiter', 'war', 'warm', 'we', 'weak', 'wednesday', 'week', 'wet', 'white', 'wide', 'wife', 'window', 'winter', 'woman', 'year', 'yellow', 'yesterday', 'you', 'young', 'youplural']

def norm(s):
    s = str(s).strip().lower()
    s = re.sub(r"^\d+\.\s*", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

# Special folder-name normalization in INCLUDE
ALIASES = {
    "second": "secondnumber",   # the recovered '81. Second' folder
}

def canonical_folder_name(name):
    n = norm(name)
    return ALIASES.get(n, n)

folders = []
for d in DATASET.rglob("*"):
    if not d.is_dir():
        continue
    videos = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    if videos:
        folders.append((d, canonical_folder_name(d.name), len(videos)))

present = set(x[1] for x in folders)
official_set = set(OFFICIAL)

missing = sorted(official_set - present)
extra = sorted(present - official_set)

print("="*78)
print("FINAL INCLUDE 263-CLASS FOLDER VERIFICATION")
print("="*78)
print(f"Unique normalized class folders : {len(present)}")
print(f"Official expected classes       : {len(official_set)}")
print(f"Missing official classes        : {len(missing)}")
print(f"Unexpected extra classes        : {len(extra)}")

if missing:
    print("\nMISSING:")
    for x in missing:
        print(" -", x)

if extra:
    print("\nEXTRA:")
    for x in extra:
        print(" -", x)

if len(present)==263 and not missing and not extra:
    print("\nALL 263 OFFICIAL INCLUDE CLASSES ARE PRESENT ✅")
else:
    print("\nVERIFICATION NEEDS REVIEW ⚠️")

print("\nRecovered target class counts:")
for target in ["clothing","pant","pocket","shoes","tshirt","secondnumber"]:
    total = sum(c for _,n,c in folders if n==target)
    print(f"{target:15s} : {total} videos")