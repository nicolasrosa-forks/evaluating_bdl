import subprocess

def launch(screen_name, command):
    subprocess.call(f"screen -dmS {screen_name} {command} &", shell=True)

if __name__ == "__main__":
    launch("nicolas_exp1_18", "python3 ensembling_train_epoch.py -M 1 -f relu -o noam -b 1 -e 5 -r 'runs/2021-11-10_20-00-23, M0, imgs=85898, relu, opt=adam, bs=1, lr=1e-05, wd=0.0005, 20/model_M0_epoch_11.pth'")

    print("Done.")
