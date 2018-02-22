# ----------------------------------------------------------------------------
#  EOgmaNeo
#  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of EOgmaNeo is licensed to you under the terms described
#  in the EOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import multiprocessing
import time
from sys import stdout
from copy import deepcopy
from random import randint
import numpy as np
import eogmaneo
import pygame
import cv2


# Restrict input image dimensions
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

WINDOW_WIDTH = int((IMAGE_WIDTH + 2) * 3.5)
WINDOW_HEIGHT = int(72 + (IMAGE_HEIGHT + 2))

# Open the example movie file, using OpenCV
# MOVIE = '../source/examples/Clock-OneArm.mp4'
MOVIE = '../source/examples/Tesseract.mp4'

cap = cv2.VideoCapture(MOVIE)

# Check if movie opened successfully
if cap.isOpened() is False:
    print("Error opening video file")

VIDEO_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
VIDEO_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

if VIDEO_WIDTH != VIDEO_HEIGHT:
    print("Video file {} has a non-square frame!".format(MOVIE))
    exit()

CAPTURE_LENGTH = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Movie has {} frames".format(CAPTURE_LENGTH))

# Mac OSX has issues with CAP_PROP_POS_FRAMES to seek to the start of a movie,
# so pre-load all the movie (and apply appropriate frame transforms)
print("Loading the movie...\n")
frames = []
for i in range(CAPTURE_LENGTH):
    ret, frame = cap.read()
    if ret is True:
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.flipud(np.rot90(frame))

        # Change from RGB to gray scale,
        # ensure range is [0, 1] (for input into the pre-encoder)
        frame = np.dot(frame[:, :, :3], [0.299, 0.587, 0.114]) / 255.0

        frames.append(frame)

# Finished with the movie now
cap.release()


# Predictive hierarchy settings
NUM_LAYERS = 4
HIDDEN_WIDTH = 32
HIDDEN_HEIGHT = 32
CHUNK_SIZE = 4
RADIUS = 9

print("Building the EOgmaNeo hierarchy and pre-encoder...\n")

# After training an ImageEncoder will output unique sparse chunked representations
# that can be input into an EOgmaNeo hierarchy. Consequently it can take predicted
# sparse chunked representations output from the hierarchy and reconstruct an
# image it has learned.
preEncoder = eogmaneo.ImageEncoder()
preEncoder.create(
    IMAGE_WIDTH, IMAGE_HEIGHT,
    HIDDEN_WIDTH, HIDDEN_HEIGHT,
    CHUNK_SIZE, int(16), 123)

# Make sure EOgmaNeo uses all CPU cores available
cs = eogmaneo.ComputeSystem(multiprocessing.cpu_count())

# Construct parameter descriptions for all hierarchy encoder-decoder layers
lds = []

for i in range(NUM_LAYERS):
    ld = eogmaneo.LayerDesc()

    ld._width = HIDDEN_WIDTH
    ld._height = HIDDEN_HEIGHT
    ld._chunkSize = CHUNK_SIZE
    ld._radius = RADIUS
    ld._ticksPerUpdate = 2
    ld._temporalHorizon = 2

    ld._alpha = 0.4
    ld._beta = 0.4

    # Disable reinforcement learning
    ld._gamma = 0.0

    lds.append(ld)

h = eogmaneo.Hierarchy()

h.create(
    [(HIDDEN_WIDTH, HIDDEN_HEIGHT)], [CHUNK_SIZE],
    [True], lds, 123)


# Initialize PyGame
pygame.init()
pygame.mixer.quit()
pygame.font.init()
pygame.display.set_caption("Video Prediction Example")

# Load a font into PyGame
font = pygame.font.SysFont('Arial', 16)

# Setup the PyGame screen
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))


def drawText(text, x, y, color=(255, 255, 255)):
    text = str(text)
    text = font.render(text, True, color)
    screen.blit(text, (x, y))


print(
    "This example shows how EOgmaNeo and the Image Pre-encoder can be used\n"
    "to predict the next image from a sequence of images.\n")

print("Step 1: Train the image pre-encoder\n"
    "  Show the image pre-encoder random frames of the video.\n"
    "  The source video image is on the left, with reconstruction\n"
    "  of that image from the pre-encoder on the right).\n")

print(
    "Step 2: Uses the trained image pre-encoder to produce sparse chunked\n"
    "  representation to send into the EOgmaNeo predictive hierarchy, and\n"
    "  step the hierarchy with training enabled. Left image shows the source video\n"
    "  frame, the right image shows the predicted video frame.\n")

print(
    "Step 3: Uses only predictions from the EOgmaNeo predictive hierarchy as input\n"
    "  to the hierarchy to predict the next video frame.\n")

print(
    "Notes: The second and last plays of the video during step 2 are slowed down!\n"
    "       The Escape key can be used to skip each step.\n")


PRE_ENCODER_ITERS = int(CAPTURE_LENGTH)
ITERATIONS = 16 #int(CAPTURE_LENGTH)

# Let the image pre-encoder see the movie more times
if "Clock-OneArm" in MOVIE:
    PRE_ENCODER_ITERS *= 4
elif "Tesseract" in MOVIE:
    PRE_ENCODER_ITERS //= 4

stdout.flush()


# --------------------------------------------------
print("Step 1: Pre-training the image pre-encoder...")

quit = False

for j in range(PRE_ENCODER_ITERS):
    if quit:
        break

    for i in range(CAPTURE_LENGTH):
        if quit:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                quit = True

        screen.fill([0, 0, 0])

        drawText("Step: {0}/{1}   Mode: Pre-encoder training".format(
            int(j + 1), PRE_ENCODER_ITERS), 12, 12)

        # Choose a random frame from the movie
        frame = deepcopy(frames[randint(0, CAPTURE_LENGTH-1)])
        # or sequentially with: frame = deepcopy(frames[i])

        # Activate the image pre-encoder and obtain a reconstructed image
        hiddenStates = preEncoder.activate(frame.ravel(), cs)
        recon = preEncoder.reconstruct(hiddenStates, cs)

        # Update the image pre-encoder weighting
        preEncoder.learn(0.95, cs)

        # Scale frame copy to [0, 255] for display
        frame *= 255.0

        # Unravel reconstructed image and scale to [0, 255] for display
        recon = np.asarray(recon).reshape((IMAGE_WIDTH, IMAGE_HEIGHT))
        recon *= 255.0

        # Copy gray scale image to RGB format for display
        s_img = np.dstack([frame.astype(np.uint8)] * 3).copy(order='C')
        r_img = np.dstack([recon.astype(np.uint8)] * 3).copy(order='C')

        # Display the original movie frame
        drawText("Source", 96 + 12, 48)
        s_surf = pygame.surfarray.make_surface(s_img)
        screen.blit(s_surf, (96, 72))

        # Display the current reconstructed image from the pre-encoder
        drawText("Reconstruction", 96 + IMAGE_WIDTH + 12, 48)
        r_surf = pygame.surfarray.make_surface(r_img)
        screen.blit(r_surf, (96 + IMAGE_WIDTH + 2, 72))

        pygame.display.update()


# --------------------------------------------------
print("Step 2: Iterating the predictive hierarchy...")

quit = False

for j in range(ITERATIONS):
    if quit:
        break

    for i in range(CAPTURE_LENGTH):
        if quit:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                quit = True

        screen.fill([0, 0, 0])

        drawText("Pass: {0} of {1}   Mode: predicting".format(
            int(j + 1), ITERATIONS), 12, 12)

        frame = deepcopy(frames[i])

        # Activate the image pre-encoder
        hiddenStates = preEncoder.activate(frame.ravel(), cs)

        # Step the predictive hierarchy
        inputs = [hiddenStates]
        h.step(inputs, cs, True)

        # Decode the predicted state
        prediction = h.getPredictions(0)
        recon = preEncoder.reconstruct(prediction, cs)

        # Scale frame copy to [0, 255] for display
        frame *= 255.0

        # Unravel reconstructed image and rescale to [0, 255] for display
        recon = np.asarray(recon).reshape((IMAGE_WIDTH, IMAGE_HEIGHT))
        recon *= 255.0

        # Copy gray scale image to RGB format for display
        s_img = np.dstack([frame.astype(np.uint8)] * 3).copy(order='C')
        r_img = np.dstack([recon.astype(np.uint8)] * 3).copy(order='C')

        # Display the original movie frame
        drawText("Source", 96 + 12, 48)
        s_surf = pygame.surfarray.make_surface(s_img)
        screen.blit(s_surf, (96, 72))

        drawText("t+0", 96 - 36, (64 + IMAGE_HEIGHT / 2))

        # Display the current reconstructed image from the pre-encoder
        drawText("Prediction", 96 + IMAGE_WIDTH + 12, 48)
        r_surf = pygame.surfarray.make_surface(r_img)
        screen.blit(r_surf, (96 + IMAGE_WIDTH + 2, 72))

        drawText("t+1", 96 + (IMAGE_WIDTH * 2) + 12, (64 + IMAGE_HEIGHT / 2))

        pygame.display.update()

        if j is 0 or j is 1:
            time.sleep(0.25)


# --------------------------------------------------
print("Step 3: Sending only t+1 predictions back into the hierarchy...")

frameCount = 0
quit = False

# for j in range(ITERATIONS):
while quit is False:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or \
           (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            quit = True

    screen.fill([0, 0, 0])

    drawText("Frame: {0}  Mode: prediction only".format(
        frameCount), 12, 12)

    frameCount += 1

    # Step the predictive hierarchy
    inputs = [h.getPredictions(0)]
    h.step(inputs, cs, False)

    # Decode the predicted state
    prediction = h.getPredictions(0)
    recon = preEncoder.reconstruct(prediction, cs)

    # Unravel reconstructed image and rescale to [0, 255] for display
    recon = np.asarray(recon).reshape((IMAGE_WIDTH, IMAGE_HEIGHT))
    recon *= 255.0

    # Convert from gray scale to RGB for display
    r_img = np.dstack([recon.astype(np.uint8)] * 3).copy(order='C')

    # Display the current reconstructed image from the pre-encoder
    drawText("Prediction", 96 + IMAGE_WIDTH + 12, 48)
    r_surf = pygame.surfarray.make_surface(r_img)
    screen.blit(r_surf, (96 + IMAGE_WIDTH + 2, 72))

    drawText("t+1", 96 + (IMAGE_WIDTH * 2) + 12, (64 + IMAGE_HEIGHT / 2))

    # 100ms delay with playback
    # time.sleep(0.1)

    pygame.display.update()

# Close PyGame (and window)
pygame.quit()
