from glob import glob
import os
from tqdm import tqdm

# filenames = glob("*/training_logs/*/checkpoint_*")
filenames = glob("*/training_logs/*/checkpoint_[!4]*")

print(filenames)
print(len(filenames))

# remove
for filename in tqdm(filenames):
    os.remove(filename)