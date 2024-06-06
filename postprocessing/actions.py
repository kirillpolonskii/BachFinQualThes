from PIL import Image

B = 0  # black px
G = 128  # gray px
W = 255  # white px
SRNDNG_G_8_B_0 = 128
SRNDNG_G_7_B_1 = 112
SRNDNG_G_0_B_8 = 0
SRNDNG_G_1_B_7 = 16
SRNDNG_G_4_W_4 = 191.5


def get3x3Grid(im: Image, i, j):  # (i, j) in center
    grid = [
        im.getpixel((i - 1, j - 1)), im.getpixel((i, j - 1)), im.getpixel((i + 1, j - 1)),
        im.getpixel((i - 1, j)), im.getpixel((i, j)), im.getpixel((i + 1, j)),
        im.getpixel((i - 1, j + 1)), im.getpixel((i, j + 1)), im.getpixel((i + 1, j + 1)),
    ]
    return grid


def get3x3GridVal(im: Image, i, j):  # (i, j) in center, it doesn't count
    val = (im.getpixel((i - 1, j - 1)) + im.getpixel((i, j - 1)) + im.getpixel((i + 1, j - 1)) +
           im.getpixel((i - 1, j)) + im.getpixel((i + 1, j)) +
           im.getpixel((i - 1, j + 1)) + im.getpixel((i, j + 1)) + im.getpixel((i + 1, j + 1)))
    return val / 8


def to3colors(img):
    for i in range(img.size[0]):  # for every pixel:
        for j in range(img.size[1]):
            if 0 <= img.getpixel((i, j)) < 11:
                img.putpixel((i, j), 0)
            elif 11 <= img.getpixel((i, j)) < 248:
                img.putpixel((i, j), 128)
            else:
                img.putpixel((i, j), 255)
    return img


def cleanBorders(img: Image):
    for i in range(img.size[0]):
        img.putpixel((0, i), 0)
        img.putpixel((1, i), 0)
        img.putpixel((i, 0), 0)
        img.putpixel((i, 1), 0)
        img.putpixel((img.size[0] - 1, i), 0)
        img.putpixel((img.size[0] - 2, i), 0)
        img.putpixel((i, img.size[0] - 1), 0)
        img.putpixel((i, img.size[0] - 2), 0)
    return img


def cleanSingleGrayPixels(img: Image):  # using
    for i in range(1, img.size[0] - 1):
        for j in range(1, img.size[1] - 1):
            if img.getpixel((i, j)) == 128:
                cur3x3GridVal = get3x3GridVal(img, i, j)
                if cur3x3GridVal == SRNDNG_G_0_B_8 or cur3x3GridVal == SRNDNG_G_1_B_7:
                    img.putpixel((i, j), 0)
    return img


def cleanSingleWhitePixels(img: Image):  # using
    for i in range(1, img.size[0] - 1):
        for j in range(1, img.size[1] - 1):
            if img.getpixel((i, j)) == 255:
                cur3x3GridVal = get3x3GridVal(img, i, j)
                if cur3x3GridVal == SRNDNG_G_8_B_0:
                    img.putpixel((i, j), 128)
    return img


def fillBlackInRooms(img: Image):  # using
    for i in range(1, img.size[0] - 1):
        for j in range(1, img.size[1] - 1):
            if img.getpixel((i, j)) == 0:
                cur3x3GridVal = get3x3GridVal(img, i, j)
                if cur3x3GridVal == SRNDNG_G_8_B_0 or cur3x3GridVal == SRNDNG_G_7_B_1:
                    img.putpixel((i, j), 128)
    return img


def replaceWhiteEntities(img: Image, num_rooms, room_difficulty):
    goalAdded = False
    cntEnemies = 0
    enemiesLimit = 0
    if room_difficulty == 0:
        enemiesLimit = 1
    elif room_difficulty == 1:
        enemiesLimit = int(num_rooms / 2)
    else:
        enemiesLimit = num_rooms
    lastEntityCoordX, lastEntityCoordY = 0, 0
    for i in range(1, img.size[0] - 1):
        for j in range(1, img.size[1] - 1):
            if img.getpixel((i, j)) == 255:
                cur3x3GridVal = get3x3GridVal(img, i, j)
                if cur3x3GridVal == SRNDNG_G_4_W_4:
                    # Clean 7 px in each direction
                    cleanSurroundingFromWhite(img, i, j)
                    # Draw "goal" if it doesn't exist
                    if not goalAdded:
                        drawGoal(img, i, j)
                        goalAdded = True
                        lastEntityCoordX, lastEntityCoordY = i, j
                    # Draw "enemy" if it doesn't exist and increase counter while it doesn't exceed limit
                    elif cntEnemies < enemiesLimit and (abs(lastEntityCoordX - i) > 7 or abs(lastEntityCoordY - j) > 7):
                        drawEnemy(img, i, j)
                        cntEnemies += 1
    return img


def cleanSurroundingFromWhite(img: Image, i, j):
    for k in range(i - 7, i + 7):
        for l in range(j - 7, j + 7):
            img.putpixel((k, l), 128)


def drawGoal(img, i, j):
    for k in range(8):
        img.putpixel((i + 3, j + k), 255)
        img.putpixel((i + 4, j + k), 255)
        img.putpixel((i + k, j + 3), 255)
        img.putpixel((i + k, j + 4), 255)


def drawEnemy(img, i, j):
    for k in range(8):
        for l in range(8):
            img.putpixel((i + k, j + l), 255)