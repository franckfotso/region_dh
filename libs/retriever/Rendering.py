# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: libs.retrieval
# Copyright (c) 2018
# Written by: Franck FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import cv2

class Rendering:

    def __init__(self, imageSize, imagesPerRow, numResults):
        # store the target image size and the number of images per row
        self.imageW = imageSize[0]
        self.imageH = imageSize[1]
        self.imagesPerRow = imagesPerRow

        # allocate memory for the output image
        numCols = numResults // imagesPerRow
        self.view = np.zeros((numCols * self.imageW, imagesPerRow * self.imageH, 3), dtype="uint8")

        # initialize the counter for the current image along with the row and column
        # number
        self.counter = 0
        self.row = 0
        self.col = 0

    def addResult(self, image, text=None, highlight=False):
        # check to see if the number of images per row has been met, and if so, reset
        # the column counter and increment the row
        if self.counter != 0 and self.counter % self.imagesPerRow == 0:
            self.col = 0
            self.row += 1

        # resize the image to the fixed width and height and set it in the view
        image = cv2.resize(image, (self.imageH, self.imageW))
        (startY, endY) = (self.row * self.imageW, (self.row + 1) * self.imageW)
        (startX, endX) = (self.col * self.imageH, (self.col + 1) * self.imageH)
        self.view[startY:endY, startX:endX] = image

        # if the text is not None, draw it
        if text is not None:
            cv2.putText(self.view, text, (startX + 10, startY + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 255), 1)

        # check to see if the result should be highlighted
        if highlight:
            cv2.rectangle(self.view, (startX + 3, startY + 3), (endX - 3, endY - 3), (0, 255, 0), 2)
        else:
            cv2.rectangle(self.view, (startX + 3, startY + 3), (endX - 3, endY - 3), (255, 0, 0), 2)

        # increment the column counter and image counter
        self.col += 1
        self.counter += 1