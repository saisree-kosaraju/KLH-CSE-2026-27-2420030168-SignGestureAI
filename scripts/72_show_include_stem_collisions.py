from pathlib import Path
import re
import urllib.request
from collections import defaultdict

ROOT = Path.cwd()
DATASET = ROOT / "datasets" / "INCLUDE"

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}

OFFICIAL_LABELS = [
"actor","adult","afternoon","alive","alright","animal","artist","attack","author","baby",
"bad","bag","ball","bank","bathroom","beautiful","bed","bedroom","bicycle","biglarge",
"bill","bird","black","blind","blue","boat","book","box","boy","brother","brown","bus",
"camera","car","card","cat","cellphone","chair","cheap","child","city","clean","clock",
"clothing","cold","colour","computer","cool","court","cow","crowd","curved","daughter",
"dead","deaf","death","deep","dirty","doctor","dog","door","dream","dress","dry",
"election","energy","evening","exercise","exmonsoon","expensive","fall","family","famous",
"fan","fast","father","female","fish","flat","friday","friend","gift","girl","god","good",
"goodafternoon","goodevening","goodmorning","goodnight","grandfather","grandmother",
"green","grey","ground","gun","happy","hard","hat","he","healthy","heavy","hello","high",
"horse","hospital","hot","hour","house","howareyou","husband","i","india","it","job",
"key","king","kitchen","lamp","laptop","lawyer","letter","library","light","location",
"lock","long","loose","loud","low","male","man","manager","market","marriage","mean",
"medicine","minute","monday","money","month","morning","mother","mouse","narrow",
"neighbour","new","newspaper","nice","night","office","old","orange","page","paint",
"pant","paper","parent","park","patient","peace","pen","pencil","photograph","pink",
"plane","player","pleased","pocket","police","poor","president","price","priest","queen",
"quiet","raceethnicity","radio","red","religion","reporter","restaurant","rich","ring",
"sad","saturday","school","science","screen","season","second","secondnumber","secretary",
"shallow","she","shirt","shoes","short","sick","sign","sister","skirt","slow",
"smalllittle","soap","soft","soldier","son","sport","spring","storeorshop","streetorroad",
"strong","student","suit","summer","sunday","table","tall","teacher","team","technology",
"telephone","television","temple","thankyou","they","thick","thin","thursday","tight",
"time","today","tomorrow","tool","train","trainstation","trainticket","transportation",
"truck","tshirt","tuesday","ugly","university","waiter","war","warm","we","weak",
"wednesday","week","wet","white","wide","wife","window","winter","woman","year","yellow",
"yesterday","you","young","youplural"
]

URLS = {
    "train": "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths/include_train.txt",
    "val": "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths/include_val.txt",
    "test": "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths/include_test.txt",
}


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def clean_class(folder):
    s = re.sub(r"^\s*\d+\s*[\.\-_:)]*\s*", "", str(folder).strip())
    return norm(s)


def stem_norm(filename):
    return norm(Path(str(filename)).stem)


def parse_split_text(text):
    text = text.replace("\r", " ").replace("\n", " ")
    pat = re.compile(r"(?i)(\.mov|\.mp4|\.avi|\.mkv|\.webm)\s+")
    text = pat.sub(lambda m: m.group(1) + "\n", text)
    return [x.strip() for x in text.splitlines() if x.strip()]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return parse_split_text(r.read().decode("utf-8"))


def main():
    print("=" * 88)
    print("INCLUDE STEP 10 - SHOW THE 3 CLASS+STEM COLLISIONS")
    print("=" * 88)

    official_groups = defaultdict(list)

    for split, url in URLS.items():
        for relpath in fetch(url):
            parts = [p for p in relpath.replace("\\", "/").split("/") if p]
            if len(parts) < 3:
                continue

            label = clean_class(parts[1])
            if label == "secondnumber":
                continue

            filename = parts[-1].strip()
            key = f"{label}|{stem_norm(filename)}"

            official_groups[key].append({
                "split": split,
                "filename": filename,
                "path": relpath,
            })

    local_groups = defaultdict(list)

    for video in DATASET.rglob("*"):
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXTS:
            continue

        rel = video.relative_to(DATASET)
        parts = rel.parts

        matches = []
        for idx in range(len(parts) - 1):
            label = clean_class(parts[idx])
            if label in OFFICIAL_LABELS:
                matches.append((idx, label))

        if not matches:
            continue

        _, label = matches[-1]
        filename = parts[-1]
        key = f"{label}|{stem_norm(filename)}"

        local_groups[key].append({
            "filename": filename,
            "path": str(rel),
        })

    collision_keys = set()

    for key, rows in official_groups.items():
        unique_official_names = {r["filename"].lower() for r in rows}
        if len(unique_official_names) > 1:
            collision_keys.add(key)

    for key, rows in local_groups.items():
        unique_local_names = {r["filename"].lower() for r in rows}
        if len(unique_local_names) > 1:
            collision_keys.add(key)

    print(f"\nCollision keys found: {len(collision_keys)}\n")

    for i, key in enumerate(sorted(collision_keys), start=1):
        print("-" * 88)
        print(f"COLLISION {i}: {key}")
        print("-" * 88)

        print("OFFICIAL:")
        for r in official_groups.get(key, []):
            print(f"  [{r['split']}] {r['path']}")

        print("\nLOCAL:")
        # Deduplicate identical extracted copies in the display.
        seen = set()
        for r in local_groups.get(key, []):
            item = (r["filename"].lower(), r["path"].lower())
            if item in seen:
                continue
            seen.add(item)
            print(f"  {r['path']}")

        print()

    print("=" * 88)
    print("Send me the COLLISION 1/2/3 blocks shown above.")
    print("Do not rename/delete any files yet.")
    print("=" * 88)


if __name__ == "__main__":
    main()