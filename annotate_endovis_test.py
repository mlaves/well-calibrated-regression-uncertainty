from matplotlib import pyplot as plt
import os
from skimage import io
from glob import glob
from fire import Fire


def annotate(start_id=0):
    plt.ion()

    base_dir = '/media/data/EndoVis15_instrument_tracking/test/'
    dataset = 'Dataset6'
    data_dir = base_dir + dataset
    img_file_names = sorted(glob(data_dir + '/*.png'))

    print("Dataset length:", len(img_file_names))

    def onclick(event):
        with open(dataset+'.csv', "a") as myfile:
            if event.button == 1:
                print(f"{int(round(event.xdata))} {int(round(event.ydata))}")
                myfile.write(f"{int(round(event.xdata))} {int(round(event.ydata))} " + filename + "\n")
            elif event.button == 3:
                print("-1 -1")
                myfile.write("-1 -1 " + filename + "\n")

    imgg = None
    fig, ax = plt.subplots()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    img_file_names = img_file_names[start_id:]
    print("Start at:   ", start_id)
    print("Still to go:", len(img_file_names))
    for filename in img_file_names:
        img = io.imread(filename)
        print(filename)
        
        if imgg is None:
            imgg = ax.imshow(img)
        else:
            imgg.set_data(img)
        
        fig.show()

        ret = plt.waitforbuttonpress(0)
        if ret:
            break


if __name__ == '__main__':
    Fire(annotate)
