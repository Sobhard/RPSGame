import pygame
import pygame_widgets
import numpy as np
import cv2
from pygame_widgets.button import Button
# from pygame_widgets.dropdown import Dropdown

# initialize model
MODELCFG = "custom-yolov4-tiny-detector.cfg"
WEIGHTS = "custom-yolov4-tiny-detector_last (5).weights"

neuralNet = cv2.dnn.readNet(MODELCFG, WEIGHTS)
neuralNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) 
neuralNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) 

layer_names = neuralNet.getLayerNames()
output_layers = [layer_names[i - 1] for i in neuralNet.getUnconnectedOutLayers()]

gesture_map = {
    0: "Paper",
    1: "Rock",
    2: "Scissors"
}

pygame.init()

def cvImage_to_pyGameSurface(frame):
    CAMERA_SIZE = int((3.8/10) * SCREEN_LENGTH)
    frame = cv2.resize(frame, (CAMERA_SIZE, CAMERA_SIZE))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_flipped = cv2.flip(frame_rgb, 0)
    frame_surface = pygame.surfarray.make_surface(np.rot90(frame_flipped, k=3))

    return frame_surface

def forwardFrame(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    neuralNet.setInput(blob)
    outputs = neuralNet.forward(output_layers)
    return outputs

def processFrame(frame, annotateFrame):
    networkOutput = forwardFrame(frame)

    height, width = frame.shape[:2]

    for out in networkOutput:
        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence < 0.3:
                continue

            if annotateFrame:
                x = int(detection[0] * width) #center
                y = int(detection[1] * height) #center
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                label = str(gesture_map[class_id])
                cv2.rectangle(frame, (x - (w//2), y - (h//2)), (x + (w//2), y + (h//2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return (gesture_map[class_id], frame)
    
    return("Nothing Detected", frame)

#really trash bad function
def toggleAnnotations():
    global show_annotations
    show_annotations = not show_annotations

def findCounter(gesture):
    if gesture == "Rock":
        return "Paper"
    elif gesture == "Paper":
        return "Scissors"
    else:
        return "Rock"

def playGame():
    global gameInProgress, gameStateIndex, titleText, displayGesture
    if not gameInProgress:
        displayGesture = True
        gameInProgress = True
        gameStateIndex = 0
        titleText = gameStateTexts[gameStateIndex]

        pygame.time.set_timer(CHANGE_GESTURE_EVENT, 1000)

SCREEN_LENGTH = 1366
SCREEN_HEIGHT = 768
LIGHTBROWN = (232, 179, 89)
BROWN = (143, 75, 2)
ORANGE = (253, 152, 50)
FONT = pygame.font.SysFont('calibri', 100)

#PNGS
rockPic = pygame.transform.scale(pygame.image.load("gesturePNGs/rock.png"), (200, 200))
paperPic = pygame.transform.scale(pygame.image.load("gesturePNGs/paper.png"), (200, 200))
scissorsPic = pygame.transform.scale(pygame.image.load("gesturePNGs/scissors.png"), (200, 200))
shootPic = pygame.transform.scale(pygame.image.load("gesturePNGs/shoot.png"), (200, 200))
backgroundPic = pygame.transform.scale(pygame.image.load("gesturePNGs/background.png"), (SCREEN_LENGTH, SCREEN_HEIGHT))

CHANGE_GESTURE_EVENT = pygame.USEREVENT + 1

#tracking game state
gameInProgress = False
gameStateIndex = 0
idleText = "Waiting"
gameStateTexts = ["ROCK", "PAPER", "SCISSOR", "SHOOT!!"]
gestureNames = ["Rock", "Paper", "Scissors","Shoot"]
titleText = idleText

show_annotations = True

#counter gesture variables
displayGesture = False
lastGesture = "Paper"
currDisplayGesture = "Scissors"

screen = pygame.display.set_mode((SCREEN_LENGTH, SCREEN_HEIGHT))
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW) 

toggleAnnotationButton = Button(
    screen,
    SCREEN_LENGTH/3 - 25, # Right X
    SCREEN_HEIGHT/5 + 400, # Top Y
    SCREEN_LENGTH/3 + 50, # Lenght
    SCREEN_HEIGHT/5 - 150, # Width

    text='Toggle Annotations',
    fontSize = 30,
    margin = 20,
    inactiveColor=(200, 50, 0),
    hoverColor=(150, 0, 0),
    pressedColor=(0, 200, 20),
    radius=20,
    onClick=lambda: toggleAnnotations()
)

startGameButton = Button(
    screen,
    SCREEN_LENGTH/3 - 25,
    SCREEN_HEIGHT/5 + 460,
    SCREEN_LENGTH/3 + 50,
    SCREEN_HEIGHT/5 - 150,

    text='Start Round',
    fontSize = 30,
    margin = 20,
    inactiveColor=(200, 50, 0),
    hoverColor=(150, 0, 0),
    pressedColor=(0, 200, 20),
    radius=20,
    onClick=lambda: playGame()
)

run = True
while run:

    events = pygame.event.get()

    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            run = False
            quit()

        if event.type == CHANGE_GESTURE_EVENT and gameInProgress:
            gameStateIndex += 1
            if gameStateIndex  < len(gameStateTexts):
                titleText = gameStateTexts[gameStateIndex]
                currDisplayGesture = gestureNames[gameStateIndex]
            else:
                pygame.time.set_timer(CHANGE_GESTURE_EVENT, 0)
                gameInProgress = False
                currDisplayGesture = findCounter(lastGesture)
                titleText = "See Below"

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gesture, annotatedFrame = processFrame(frame, show_annotations)
    lastGesture = gesture

    screen.fill(LIGHTBROWN)
    screen.blit(backgroundPic, (0,0))

    surface = cvImage_to_pyGameSurface(annotatedFrame)
    surface_width, surface_height = surface.get_size()
    
    #border
    #pygame.draw.rect(screen, BROWN, (SCREEN_LENGTH/3 - 25, SCREEN_HEIGHT/5 - 25, surface_width + 50, surface_height + 50), width=25, border_radius=25)

    #ROCK PAPER SCISSOR SHOOT display/text box
    pygame.draw.rect(screen, ORANGE, pygame.Rect(SCREEN_LENGTH/3 - 25, 20, 580, 150), border_radius=25)
    screen.blit(FONT.render(titleText, True, BROWN), (SCREEN_LENGTH/3 + 50, 50))

    screen.blit(surface, (int((0.77/10) * SCREEN_LENGTH), int((0.97 / 5.63) * SCREEN_HEIGHT)))

    if displayGesture:
        x = 700
        y = 750
        if currDisplayGesture == "Rock":
            screen.blit(rockPic, (x, y))
        if currDisplayGesture == "Paper":
            screen.blit(paperPic, (x, y))
        if currDisplayGesture == "Scissors":
            screen.blit(scissorsPic, (x, y))
        if currDisplayGesture == "Shoot":
            screen.blit(shootPic, (x, y))

    
    pygame_widgets.update(events)
    pygame.display.update()