import os
for split in ['train', 'val', 'test']:
    for cls in os.listdir(os.path.join('dataset', split)):
        print(f"{split}/{cls}: {len(os.listdir(os.path.join('dataset', split, cls)))} images")
