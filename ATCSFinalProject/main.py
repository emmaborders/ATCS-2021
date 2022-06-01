import numpy as np  # storing and manipulating data
import pandas as pd
import matplotlib.pyplot as plt  # plotting
from sklearn.cluster import KMeans  # ML
from sklearn.linear_model import LogisticRegression
import matplotlib.image as mpimg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import os.path


class ImageCompressor:
    def __init__(self):
        self.imgSize = None

    def loadImage(self, filepath):
        """
        Uses the file path to load the image.
        Stores it as a 2D array of red-green-blue values.
        This function takes in a file path and returns
        a 2D list of [red, blue, green] values.
        """
        # Reads image
        img = mpimg.imread(filepath)
        self.imgSize = img.shape
        return img


    def saveImage(self, img, filepath):
        """
        Saves the image (currently stored as a 2D list) to the file
        path given as a parameter.
        Takes in a 2D array and a file path.
        """
        mpimg.imsave(filepath, img)

    def convertTo1D(self, img):
        """
        Converts the 2D array of [Red,Green,Blue] values to a 1D array of [Red,Green,Blue] values.
        """
        return img.reshape(self.imgSize[0] * self.imgSize[1], self.imgSize[2])

    def convertTo2D(self, img):
        """
        Converts the 1D array of [Red,Green,Blue] values to a 2D array of [Red,Green,Blue] values.
        Takes in a 1D array representing an image and returns it as a 2D List.
        """
        img = np.clip(img.astype('uint8'), 0, 255)
        img = img.reshape(self.imgSize[0], self.imgSize[1], self.imgSize[2])
        return img

    def plotImageComparisons(self, original, compressed):
        """
        Plots two images, each represented by a 2D array, on the same figure.
        One image is the original, the other is the edited image.
        """
        fig, ax = plt.subplots(1, 2)

        # original image
        ax[0].imshow(original)
        ax[0].set_title('Original Image')

        # edited image
        ax[1].imshow(compressed)
        ax[1].set_title('Compressed Image')

        # removes axes and "draws" the figure
        for ax in fig.axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def plotImageColors(self, img):
        """
        Plots the pixels on a scatter plot.
        Uses a 3D plot with axes representing red, green and blue.
        Creates the scatter plot using the 2D array passed into the function.
        """
        fig = plt.figure()
        ax = fig.addSubplot(111, projection='3d')

        ax.setXLabel("Red")
        ax.setYLabel("Green")
        ax.setZLabel("Blue")

        ax.scatter(img[:, 0], img[:, 1], img[:, 2], c=img / 255.0)
        plt.show()

    def changeColorsBlue(self, averageColors):
        """
        Reassigns the average colors (parameter) to their corresponding shades of blue.
        This is the first of 7 functions that takes in an array of red-blue-green values
        and returns the array once its values have been changed.
        """

        # for each group of pixels that have been assigned to their average color
        for i in range(len(averageColors)):

            # assigns a number to the rgb value to represent it's "brightness"
            # Someone created a way of calculating a color's relative lightness/darkness
            # https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
            brightness = round((averageColors[i][0] * 0.2126) + (averageColors[i][1] * 0.7152) + (averageColors[i][2] * 0.0722))

            # two types of blue shades
            # dark blues: red and green are zero, blue varies
            # light blues: red and green values vary but remain equal to each other, blue is constant at 255
            # if else statement determines if the new color will be light or dark blue and assigns it to a shade of blue accordingly

            if brightness < 128:
                blueValue = (brightness * 2)
                averageColors[i] = [0, 0, blueValue]

            else:
                redGreenValue = ((brightness - 128) * 2)
                averageColors[i] = [redGreenValue, redGreenValue, 255]

    def changeColorsRed(self, averageColors):
        """
        Reassigns the average colors (parameter) to their corresponding shades of red.
        works the same was as the "changeColorsBlue" function above.
        """
        for i in range(len(averageColors)):

            brightness = round((averageColors[i][0] * 0.2126) + (averageColors[i][1] * 0.7152) + (averageColors[i][2] * 0.0722))

            if brightness < 128:
                redValue = (brightness * 2)
                averageColors[i] = [redValue, 0, 0]

            else:
                greenBlueValue = ((brightness - 128) * 2)
                averageColors[i] = [255, greenBlueValue, greenBlueValue]

    def changeColorsGreen(self, averageColors):
        """
        Reassigns the average colors (parameter) to their corresponding shades of green.
        works the same was as the "changeColorsBlue" and "changeColorRed" functions above.
        """
        for i in range(len(averageColors)):

            brightness = round((averageColors[i][0] * 0.2126) + (averageColors[i][1] * 0.7152) + (averageColors[i][2] * 0.0722))

            if brightness < 128:
                greenValue = (brightness * 2)
                averageColors[i] = [0, greenValue, 0]

            else:
                redBlueValue = ((brightness - 128) * 2)
                averageColors[i] = [redBlueValue, 255, redBlueValue]

    def changeColorsChristmas(self, averageColors):
        """
        Reassigns each of the average colors to a corresponding christmas themed color.
        Uses KNN to pick the closest color.
        reads in a csv of the six color options.
        """

        # this variable ensures that each color will only be used once
        # that way the final image uses each of the christmas colors
        skipRows = []

        # for each value in averageColors
        for i in range(len(averageColors)):

            # reads csv excluding the rows of colors that have already been used
            data = pd.read_csv("data/christmas.csv", skiprows=skipRows)

            colorRed = data["red"].values
            colorGreen = data["green"].values
            colorBlue = data["blue"].values
            color = data["color"].values

            # stores the red-blue-green values from the csv into a 2D array
            colors = np.array([colorRed, colorGreen, colorBlue]).transpose()

            # standardizes data
            scaler = StandardScaler().fit(colors)
            colors = scaler.transform(colors)

            # creates knn model with a nearest neighbor of 1 to pick "newColor"
            model = KNeighborsClassifier(n_neighbors=1).fit(colors, color)
            newColor = model.predict([averageColors[i]])

            # if-statements reassign the average color at index i to the rgb of "newColor"
            # and add the row of "newColor" to skipRows
            if newColor == 'cinnamon':
                averageColors[i] = [201, 116, 71]
                skipRows.append(1)

            if newColor == 'crimson':
                averageColors[i] = [133, 13, 11]
                skipRows.append(2)

            if newColor == 'berry':
                averageColors[i] = [196, 18, 65]
                skipRows.append(3)

            if newColor == 'mistletoe':
                averageColors[i] = [111, 122, 77]
                skipRows.append(4)

            if newColor == 'wreath':
                averageColors[i] = [60, 69, 53]
                skipRows.append(5)

            if newColor == 'rose':
                averageColors[i] = [255, 240, 243]
                skipRows.append(6)


    def changeColorsMiamiVice(self, averageColors):
        """
        Reassigns each of the average colors to a corresponding miami vice themed color.
        identical to the changeColorsChristmas function but for a different color palette.
        """

        skipRows = []

        for i in range(len(averageColors)):

            data = pd.read_csv("data/miamiVice.csv", skiprows=skipRows)

            colorRed = data["red"].values
            colorGreen = data["green"].values
            colorBlue = data["blue"].values
            color = data["color"].values

            colors = np.array([colorRed, colorGreen, colorBlue]).transpose()

            scaler = StandardScaler().fit(colors)
            colors = scaler.transform(colors)

            model = KNeighborsClassifier(n_neighbors=1).fit(colors, color)
            newColor = model.predict([averageColors[i]])

            if newColor == 'turquoise':
                averageColors[i] = [85, 242, 240]
                skipRows.append(1)

            if newColor == 'pink':
                averageColors[i] = [255,56,219]
                skipRows.append(2)

            if newColor == 'blueberry':
                averageColors[i] = [56 ,106 ,255]
                skipRows.append(3)

            if newColor == 'navy':
                averageColors[i] = [38, 18, 138]
                skipRows.append(4)

            if newColor == 'violet':
                averageColors[i] = [184, 102, 250]
                skipRows.append(5)


    def changeColorsSherbet(self, averageColors):
        """
        Reassigns each of the average colors to a corresponding sherbet themed color.
        identical to the changeColorsChristmas function but for a different color palette.
        """

        skipRows = []

        for i in range(len(averageColors)):
            data = pd.read_csv("data/sherbet.csv", skiprows=skipRows)

            colorRed = data["red"].values
            colorGreen = data["green"].values
            colorBlue = data["blue"].values
            color = data["color"].values

            colors = np.array([colorRed, colorGreen, colorBlue]).transpose()

            scaler = StandardScaler().fit(colors)
            colors = scaler.transform(colors)

            model = KNeighborsClassifier(n_neighbors=1).fit(colors, color)
            newColor = model.predict([averageColors[i]])

            if newColor == 'orange':
                averageColors[i] = [255, 145, 10]
                skipRows.append(1)

            if newColor == 'sunshine':
                averageColors[i] = [255, 189, 46]
                skipRows.append(2)

            if newColor == 'peach':
                averageColors[i] = [255, 216, 186]
                skipRows.append(3)

            if newColor == 'flamingo':
                averageColors[i] = [255, 169, 204]
                skipRows.append(4)

            if newColor == 'fuchsia':
                averageColors[i] = [255, 92, 159]
                skipRows.append(5)

            if newColor == 'yellow':
                averageColors[i] = [255, 234, 94]
                skipRows.append(6)


    def changeColors(self, averageColors):
        """
        Reassigns each of the average colors to a corresponding color from the "rgb" csv.
        Instead of using KNN, reads in a csv with lots of data and graphs the rbg values
        from the csv, using said values as training data.

        """

        # "usedColors" is needed to determine which rows to skip
        usedColors = []
        skipRows = []

        for i in range(len(averageColors)):

            data = pd.read_csv("data/rgb.csv", skiprows=skipRows)

            # red, green, blue values are assigned to x
            # corresponding palette colors are assigned to y
            x = data[["red", "green", "blue"]].values
            y = data["color"].values

            # standardizes data
            scaler = StandardScaler().fit(x)
            x = scaler.transform(x)

            model = LogisticRegression().fit(x, y)

            # assigns variables to the red-blue-green value of the average color at i
            red = averageColors[i][0]
            green = averageColors[i][1]
            blue = averageColors[i][2]

            # makes prediction
            xPred = [[red, green, blue]]
            xPred = scaler.transform(xPred)

            # assigns newColor to prediction
            newColor = model.predict(xPred)[0]

            # If-statements change the rbg value of "averageColors" at index i
            # add palette color to "usedColors"
            # parse through the "rbg" csv to look for the palette color and add its rows to "skipRows"
            if newColor == 'raspberry':
                averageColors[i] = [207, 48, 95]
                usedColors.append('raspberry')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'raspberry':
                        skipRows.append(j + 1)

            if newColor == 'cherry':
                averageColors[i] = [250, 110, 110]
                usedColors.append('cherry')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'cherry':
                        skipRows.append(j + 1)

            if newColor == 'watermelon':
                averageColors[i] = [255, 133, 175]
                usedColors.append('watermelon')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'watermelon':
                        skipRows.append(j + 1)

            if newColor == 'bubblegum':
                averageColors[i] = [255, 194, 215]
                usedColors.append('bubblegum')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'bubblegum':
                        skipRows.append(j + 1)

            if newColor == 'peach':
                averageColors[i] = [255, 182, 173]
                usedColors.append('peach')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'peach':
                        skipRows.append(j + 1)

            if newColor == 'tangerine':
                averageColors[i] = [255, 175, 56]
                usedColors.append('tangerine')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'tangerine':
                        skipRows.append(j + 1)

            if newColor == 'gold':
                averageColors[i] = [255, 221, 84]
                usedColors.append('gold')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'gold':
                        skipRows.append(j+1)

            if newColor == 'banana':
                averageColors[i] = [255, 244, 150]
                usedColors.append('banana')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'banana':
                        skipRows.append(j+1)

            if newColor == 'mint':
                averageColors[i] = [219, 255, 148]
                usedColors.append('mint')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'mint':
                        skipRows.append(j+1)

            if newColor == 'lime':
                averageColors[i] = [174, 237, 126]
                usedColors.append('lime')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'lime':
                        skipRows.append(j+1)

            if newColor == 'teal':
                averageColors[i] = [102, 227, 148]
                usedColors.append('teal')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'teal':
                        skipRows.append(j+1)

            if newColor == 'forest':
                averageColors[i] = [119, 207, 93]
                usedColors.append('forest')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'forest':
                        skipRows.append(j+1)

            if newColor == 'ice':
                averageColors[i] = [191, 255, 250]
                usedColors.append('ice')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'ice':
                        skipRows.append(j+1)

            if newColor == 'winter':
                averageColors[i] = [95, 230, 237]
                usedColors.append('winter')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'winter':
                        skipRows.append(j+1)

            if newColor == 'sky':
                averageColors[i] = [52, 184, 237]
                usedColors.append('sky')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'sky':
                        skipRows.append(j+1)

            if newColor == 'royal':
                averageColors[i] = [92, 137, 219]
                usedColors.append('royal')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'royal':
                        skipRows.append(j+1)

            if newColor == 'lilac':
                averageColors[i] = [207, 205, 250]
                usedColors.append('lilac')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'lilac':
                        skipRows.append(j+1)

            if newColor == 'violet':
                averageColors[i] = [212, 181, 245]
                usedColors.append('violet')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'violet':
                        skipRows.append(j+1)

            if newColor == 'berry':
                averageColors[i] = [154, 120, 173]
                usedColors.append('berry')
                df = pd.read_csv("data/rgb.csv")
                for j in range(len(df)):
                    if df.iloc[j, 3] == 'berry':
                        skipRows.append(j+1)


    def compressImage(self, img, num):

        """
        Uses KMeans clustering to group the rgb values of the 2D array (parameter) into average colors.
        Creates a list of average colors (centroids).
        Uses one of seven functions to change the centroids into different colors.
        takes in a number input by the user to determine which function will edit the centroids.
        Returns a 2D array representing the edited image.
        """

        # assigns k (the number of colors in the final image) depending on which theme the user chose
        if num == 1 or num == 6:
            k = 5
        if num == 2 or num == 3 or num == 4:
            k = 15
        if num == 5 or num == 7:
            k = 6

        # Converts 2D to 1D
        image1D = self.convertTo1D(img)

        # uses k means to create the centroids
        km = KMeans(n_clusters=k).fit(image1D)
        centroids = km.cluster_centers_

        # Calls the function that corresponds to the user's input
        if num == 1:
            self.changeColors(centroids)
        if num == 2:
            self.changeColorsBlue(centroids)
        if num == 3:
            self.changeColorsRed(centroids)
        if num == 4:
            self.changeColorsGreen(centroids)
        if num == 5:
            self.changeColorsChristmas(centroids)
        if num == 6:
            self.changeColorsMiamiVice(centroids)
        if num == 7:
            self.changeColorsSherbet(centroids)

        labels = km.labels_

        # assigns each pixel to its edited, average color
        for i in range(len(image1D)):
            image1D[i] = centroids[labels[i]]

        # Coverts back to 2D and returns
        finalImage  = self.convertTo2D(image1D)
        return finalImage



if __name__ == '__main__':
    """
    UI that allows user to choose a jpeg file from the computer's desktop and choose a color theme.
    Calls functions to create the new image and plot a side by side comparison of the edited and unedited images.
    """

    # Create image compressor object
    imageComp = ImageCompressor()

    # Prompt the user for input
    print("Enter the name of a jpeg file on this computer's desktop")
    fileName = input('example: for photo.jpeg, enter "photo"\n')

    # found a way to check if the file exists (user can't break the program)
    # https://www.pythontutorial.net/python-basics/python-check-if-file-exists/
    file_exists = os.path.exists("/Users/emmaborders/Desktop/" + fileName + ".jpeg")

    # Continues prompting for a file name until input is valid
    while file_exists == False:
        fileName = input('No such file exists, enter a different file name: ')
        file_exists = os.path.exists("/Users/emmaborders/Desktop/" + fileName + ".jpeg")

    # gives theme options
    print("There are several theme options for editing your photo:")
    print("1 = basic pop art colors")
    print("2 = blue")
    print("3 = red")
    print("4 = green")
    print("5 = christmas themed colors")
    print("6 = miami vice themed colors")
    print("7 = sherbet themed colors")

    # Prompts the user to pick a theme
    number = input("Enter the number corresponding to the theme of your choosing: ")

    # Ensures that the user can't break the code
    while number.isdigit() == False:
        number = input("That is not an integer, enter a number: ")

    colorNumber = int(number)

    while colorNumber not in {1, 2, 3, 4, 5, 6, 7}:
        number = input("That does not correspond to a theme, enter a number 1 through 7: ")

        while number.isdigit() == False:
            number = input("That is not an integer, enter a number: ")

        colorNumber = int(number)

    # Loads image by making a call to the "loadImage" function
    image = imageComp.loadImage("/Users/emmaborders/Desktop/" + fileName + ".jpeg")

    # Calls the "compressImage" function to create the edited image
    newImage = imageComp.compressImage(image, colorNumber)
    # plots the comparison figure
    imageComp.plotImageComparisons(imageComp.loadImage("/Users/emmaborders/Desktop/" + fileName + ".jpeg"), newImage)

