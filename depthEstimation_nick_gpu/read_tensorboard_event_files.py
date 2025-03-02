# https://stackoverflow.com/questions/41074688/how-do-you-read-tensorboard-files-programmatically
from tensorboard.backend.event_processing import event_accumulator
from glob import glob
import pathlib
from send2trash import send2trash


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def main():

    files = glob("./runs/*/events.out.tfevents*")
    files.sort()  # Sort by

    print(f"Num of 'runs' found: {len(files)}")

    delete_it = False

    for file in files:
        try:
            ea = event_accumulator.EventAccumulator(file, size_guidance={event_accumulator.SCALARS: 0,})
            ea.Reload()
            
            # print(ea.Tags())
            try:
                trained_steps = len(ea.Scalars('Train/Loss'))
            except KeyError:
                trained_steps = len(ea.Scalars('Train/Loss_v2'))

            if trained_steps >= 10000:
                color = bcolors.OKGREEN
                delete_it = False
            else:
                color = bcolors.WARNING
                delete_it = True

            print(f"{color}{file}\t{trained_steps}{bcolors.ENDC}")
            
            if delete_it:
                raise KeyError
        
        except KeyError:  # Doesn't have the 'Train/Loss' or 'Train/Loss_v2' tags, then delete it.
            parent_folder = pathlib.Path(file).parent

            # print(file)
            print(f"Deleting '{parent_folder}' ...")

            send2trash(str(parent_folder))

    print("\nDone.")

if __name__ == '__main__':
    main()