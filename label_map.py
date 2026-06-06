import pickle
import os

def load_label_map(path='label_map.pkl'):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        CHAR_TO_IDX = data.get('CHAR_TO_IDX')
        IDX_TO_CHAR = data.get('IDX_TO_CHAR')
        NUM_CLASSES = data.get('NUM_CLASSES')
        all_classes = data.get('ALL_CLASSES')

        print(f"✅ Loaded label map ({NUM_CLASSES} classes)")
    else:
        all_classes = [
            '0','1','2','3','4','5','6','7','8','9',
            'A','B','C','D','E','F','G','H','J','K',
            'L','M','N','P','R','S','T','Z'
        ]

        CHAR_TO_IDX = {ch: i for i, ch in enumerate(all_classes)}
        IDX_TO_CHAR = {i: ch for i, ch in enumerate(all_classes)}
        NUM_CLASSES = len(all_classes)

        print("⚠ Using fallback label map")

    return CHAR_TO_IDX, IDX_TO_CHAR, NUM_CLASSES, all_classes


def decode_prediction(idx, IDX_TO_CHAR):
    return IDX_TO_CHAR.get(idx, '?')