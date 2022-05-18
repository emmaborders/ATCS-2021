import numpy as np  # storing and manipulating data
import pandas
import pandas as pd
import matplotlib.pyplot as plt  # plotting
from sklearn.cluster import KMeans  # ML
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle  # ML
from tkinter import *
from tkinter import ttk
import PIL  # opening image and converting back from array
import matplotlib.image as mpimg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score



class ImageCompressor:
    def __init__(self):
        self.img_size = None

class ImageCompressor:


    def __init__(self):
        self.img_size = None

    def load_image(self, filepath):
        """
        Loads the image from the path as a 2D List (Height x Width) of [R,G,B] values.
        :param filepath: Path to the original image
        :return: 2D List of [R, G, B] values
        """
        # Read the image
        img = mpimg.imread(filepath)
        self.img_size = img.shape

        return img

    def save_image(self, img, filepath):
        """
        Saves the provided image to the file specified by filepath
        :param img: 2D List of [R, G, B] values that represent the image
        :param filepath: Location to save the file
        """
        mpimg.imsave(filepath, img)

    def convert_to_1D(self, img):
        """
        Converts a 2D List of [R,G,B] values into a 1D List of [R,G,B] values
        :param img: 2D List of [R,G,B] values that represent an image
        :return: 1D List of [R,G,B] values that represent an image
        """
        return img.reshape(self.img_size[0] * self.img_size[1], self.img_size[2])

    def convert_to_2D(self, img):
        """
        Converts a 1D List of pixels where each pixel is represented by [R,G,B] into
        a 2D List of dimensions height x width where each entry is a [R,G,B] pixel
        :param img: 1D List of [R,G,B] values for each pixel
        :return: 2D list of dimensions height x width where each entry is an [R,G,B] pixel
        """
        img = np.clip(img.astype('uint8'), 0, 255)
        img = img.reshape(self.img_size[0], self.img_size[1], self.img_size[2])
        return img

    def plot_image_comparisons(self, original, compressed):
        """
        Plots the original and compressed image on the same figure
        :param original: 2D List of [R,G,B] values representing the original image
        :param compressed: 2D List of [R,G,B] values representing the compressed image
        """
        fig, ax = plt.subplots(1, 2)

        # Plot the original image
        ax[0].imshow(original)
        ax[0].set_title('Original Image')

        # Plot the compressed image
        ax[1].imshow(compressed)
        ax[1].set_title('Compressed Image')

        # Turn the axes off and show the figure
        for ax in fig.axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_image_colors(self, img):
        """
        Plots the colors in an image on a 3D scatter plot
        :param img: A 2D List of pixels where each pixel is represented by [R,G,B]
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel("Red")
        ax.set_ylabel("Green")
        ax.set_zlabel("Blue")

        ax.scatter(img[:, 0], img[:, 1], img[:, 2], c=img / 255.0)
        plt.show()

    def change_colors(self, average_colors):

        used_colors = []
        skip_rows = []

        for i in range(len(average_colors)):

            print(skip_rows)
            data = pd.read_csv("data/rgb.csv", skiprows=skip_rows)


            x = data[["red", "green", "blue"]].values
            y = data["color"].values


            scaler = StandardScaler().fit(x)
            x = scaler.transform(x)


            model = LogisticRegression().fit(x, y)

            coef = model.coef_[0]

            ''' Make a new prediction '''
            red = average_colors[i][0]
            green = average_colors[i][1]
            blue = average_colors[i][2]

            x_pred = [[red, green, blue]]
            x_pred = scaler.transform(x_pred)

            new_color = model.predict(x_pred)[0]

            if new_color == 'raspberry':
                average_colors[i] = [207, 48, 95]
                used_colors.append('raspberry')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'raspberry':
                        skip_rows.append(j+1)


            if new_color == 'cherry':
                average_colors[i] = [250, 110, 110]
                used_colors.append('cherry')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'cherry':
                        skip_rows.append(j+1)


            if new_color == 'watermelon':
                average_colors[i] = [255, 133, 175]
                used_colors.append('watermelon')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'watermelon':
                        skip_rows.append(j+1)


            if new_color == 'bubblegum':
                average_colors[i] = [255, 194, 215]
                used_colors.append('bubblegum')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'bubblegum':
                        skip_rows.append(j+1)


            if new_color == 'peach':
                average_colors[i] = [255, 182, 173]
                used_colors.append('peach')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'peach':
                        skip_rows.append(j+1)


            if new_color == 'tangerine':
                average_colors[i] = [255, 175, 56]
                used_colors.append('tangerine')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'tangerine':
                        skip_rows.append(j+1)


            if new_color == 'gold':
                average_colors[i] = [255, 221, 84]
                used_colors.append('gold')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'gold':
                        skip_rows.append(j+1)


            if new_color == 'banana':
                average_colors[i] = [255, 244, 150]
                used_colors.append('banana')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'banana':
                        skip_rows.append(j+1)


            if new_color == 'mint':
                average_colors[i] = [219, 255, 148]
                used_colors.append('mint')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'mint':
                        skip_rows.append(j+1)


            if new_color == 'lime':
                average_colors[i] = [174, 237, 126]
                used_colors.append('lime')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'lime':
                        skip_rows.append(j+1)


            if new_color == 'teal':
                average_colors[i] = [102, 227, 148]
                used_colors.append('teal')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'teal':
                        skip_rows.append(j+1)


            if new_color == 'forest':
                average_colors[i] = [119, 207, 93]
                used_colors.append('forest')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'forest':
                        skip_rows.append(j+1)


            if new_color == 'ice':
                average_colors[i] = [191, 255, 250]
                used_colors.append('ice')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'ice':
                        skip_rows.append(j+1)


            if new_color == 'winter':
                average_colors[i] = [95, 230, 237]
                used_colors.append('winter')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'winter':
                        skip_rows.append(j+1)


            if new_color == 'sky':
                average_colors[i] = [52, 184, 237]
                used_colors.append('sky')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'sky':
                        skip_rows.append(j+1)


            if new_color == 'royal':
                average_colors[i] = [92, 137, 219]
                used_colors.append('royal')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'royal':
                        skip_rows.append(j+1)


            if new_color == 'lilac':
                average_colors[i] = [207, 205, 250]
                used_colors.append('lilac')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'lilac':
                        skip_rows.append(j+1)


            if new_color == 'violet':
                average_colors[i] = [212, 181, 245]
                used_colors.append('violet')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'violet':
                        skip_rows.append(j+1)


            if new_color == 'berry':
                average_colors[i] = [154, 120, 173]
                used_colors.append('berry')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'berry':
                        skip_rows.append(j+1)



        print(used_colors)





    def compress_image(self, img):

        """
        Compresses the image using KMeans clustering to contain
        only a set number of colors
        :param img: A 2D List of [R, G, B] values representing the image
        :return: A 2D List of [R, G, B] values representing the compressed image
        """

        image1D = self.convert_to_1D(img)

        k = 15

        km = KMeans(n_clusters=k).fit(image1D)
        centroids = km.cluster_centers_
        #print(centroids)

        self.change_colors(centroids)


        labels = km.labels_
        #print(labels)

        for i in range(len(image1D)):
            image1D[i] = centroids[labels[i]]

        finalImage  = self.convert_to_2D(image1D)

        return finalImage




if __name__ == '__main__':
    imageComp = ImageCompressor()
    image = imageComp.load_image("/Users/emmaborders/Desktop/tiger.jpeg")

    newImage = imageComp.compress_image(image)
    imageComp.plot_image_comparisons( imageComp.load_image("/Users/emmaborders/Desktop/tiger.jpeg"), newImage)

