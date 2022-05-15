import argparse
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input image")
    parser.add_argument("--csv", help="best keypoints csv")
    args = parser.parse_args()

    img = plt.imread(args.input)
    filtered_coords = pd.read_csv(args.csv)

    plt.figure(figsize=(16,16))
    plt.imshow(img, cmap='gray')
    plt.plot(filtered_coords.iloc[:,1], filtered_coords.iloc[:,0], 'x', c='r')
    plt.axis("off")
    plt.savefig("foo.png")