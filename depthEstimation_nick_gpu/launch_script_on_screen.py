import subprocess
import time

def launch(screen_name, command):
    print(f"{screen_name} | {command}")
    subprocess.call(f"screen -dmS {screen_name} {command} &", shell=True)

if __name__ == "__main__":
    print("Launching...")

    # launch("test", "python3 test.py")
    # launch("tensorboard", "./run_tensorboard.sh")

    # (running) launch("nicolas_exp1_14", "python3 ensembling_train_epoch.py -M 1 -f relu -o adam -b 1 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o adam -b 2 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o adam -b 4 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o adam -b 6 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o adam -b 8 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o adam -b 10 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o adam -b 12 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o adam -b 14 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")

    # (running) launch("nicolas_exp1_23", "python3 ensembling_train_epoch.py -M 1 -f relu -o adabelief -b 1 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o adabelief -b 2 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o adabelief -b 4 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o adabelief -b 6 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o adabelief -b 8 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o adabelief -b 10 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o adabelief -b 12 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o adabelief -b 14 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")

    # (running) launch("nicolas_exp1_32", "python3 ensembling_train_epoch.py -M 1 -f relu -o noam -b 1 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o noam -b 2 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o noam -b 4 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o noam -b 6 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o noam -b 8 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o noam -b 10 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f relu -o noam -b 12 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")
    # launch("nicolas_exp1_39", "python3 ensembling_train_epoch.py -M 1 -f relu -o noam -b 14 -e 5 -r 'runs/2021-11-14_21-07-32, M0, imgs=85898, relu, opt=adam, bs=2, lr=1e-05, wd=0.0005, 5/model_M0_epoch_5.pth'")

    # launch("nicolas_exp1_41", "python3 ensembling_train_epoch.py -M 1 -f elu -o adam -b 1 -e 5 -r 'runs/2021-11-19_15-38-48, M0, imgs=78982, elu, opt=noam, bs=1, lr=0, wd=0.0005, 5/model_M0_epoch_1.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o adam -b 2 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o adam -b 4 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o adam -b 6 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o adam -b 8 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o adam -b 10 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o adam -b 12 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o adam -b 14 -e 5")

    # launch("nicolas_exp1_50", "python3 ensembling_train_epoch.py -M 1 -f elu -o adabelief -b 1 -e 5 -r 'runs/2021-11-19_15-38-48, M0, imgs=78982, elu, opt=noam, bs=1, lr=0, wd=0.0005, 5/model_M0_epoch_1.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o adabelief -b 2 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o adabelief -b 4 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o adabelief -b 6 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o adabelief -b 8 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o adabelief -b 10 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o adabelief -b 12 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o adabelief -b 14 -e 5")

    # (running) launch("nicolas_exp1_59", "python3 ensembling_train_epoch.py -M 1 -f elu -o noam -b 1 -e 5 -r 'runs/2021-11-19_15-38-48, M0, imgs=78982, elu, opt=noam, bs=1, lr=0, wd=0.0005, 5/model_M0_epoch_1.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o noam -b 2 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o noam -b 4 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o noam -b 6 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o noam -b 8 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o noam -b 10 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o noam -b 12 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f elu -o noam -b 14 -e 5")

    # launch("nicolas_exp1_68", "python3 ensembling_train_epoch.py -M 1 -f selu -o adam -b 1 -e 5 -r 'runs/2021-11-15_20-16-34, M0, imgs=85898, selu, opt=adam, bs=1, lr=1e-05, wd=0.0005, 5/model_M0_epoch_2.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o adam -b 2 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o adam -b 4 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o adam -b 6 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o adam -b 8 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o adam -b 10 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o adam -b 12 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o adam -b 14 -e 5")

    # launch("nicolas_exp1_77", "python3 ensembling_train_epoch.py -M 1 -f selu -o adabelief -b 1 -e 5 -r 'runs/2021-11-15_20-16-34, M0, imgs=85898, selu, opt=adam, bs=1, lr=1e-05, wd=0.0005, 5/model_M0_epoch_2.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o adabelief -b 2 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o adabelief -b 4 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o adabelief -b 6 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o adabelief -b 8 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o adabelief -b 10 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o adabelief -b 12 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o adabelief -b 14 -e 5")

    # launch("nicolas_exp1_86", "python3 ensembling_train_epoch.py -M 1 -f selu -o noam -b 1 -e 5 -r 'runs/2021-11-15_20-16-34, M0, imgs=85898, selu, opt=adam, bs=1, lr=1e-05, wd=0.0005, 5/model_M0_epoch_2.pth'")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o noam -b 2 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o noam -b 4 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o noam -b 6 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o noam -b 8 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o noam -b 10 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o noam -b 12 -e 5")
    # launch("nicolas_exp1_X", "python3 ensembling_train_epoch.py -M 1 -f selu -o noam -b 14 -e 5")

    print("Done.\n")
    time.sleep(1)
    subprocess.call("screen -ls", shell=True)
