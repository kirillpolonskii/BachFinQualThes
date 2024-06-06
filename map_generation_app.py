from io import BytesIO

import numpy as np
import streamlit as st
import torch
from PIL import Image

from cgan.model import Discriminator, Generator
from cgan.utils import load_checkpoint
from postprocessing.actions import to3colors, cleanSingleGrayPixels, cleanSingleWhitePixels, fillBlackInRooms, \
    cleanBorders, replaceWhiteEntities

# Hyperparameters
device = "cpu"
print(device)
LEARNING_RATE = 0.0007
BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS_IMG = 1
NUM_CLASSES = 30
GEN_EMBEDDING = 140
Z_DIM = 140
NUM_EPOCHS = 20
FEATURES_CRITIC = 48
FEATURES_GEN = 48
CRITIC_ITERATIONS = 4
LAMBDA_GP = 10
LOAD_CHECKPOINT = False
GRID_SIZE = 4

st.set_page_config(layout="wide")
st.markdown("""
            <style>
                div[data-testid="column"] {
                    width: fit-content !important;
                    flex: unset;
                }
                div[data-testid="column"] * {
                    width: fit-content !important;
                }
            </style>
            """, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Приложение для генерации одностраничных карт</h1>",
            unsafe_allow_html=True)
pg_bg = """
<style>
[data-testid="stAppViewContainer"] {
background-color: #e6e6e6;
}
</style>
"""
st.markdown(pg_bg, unsafe_allow_html=True)
col = st.columns([0.25, 0.25, 0.25, 0.25], gap="small")
num_rooms = col[0].selectbox('Количество комнат:', options=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
room_difficulty = col[1].selectbox('Уровень сложности:', options=['Лёгкий', 'Средний', 'Тяжёлый'])
btn_generate = col[2].button(
        label="Сгенерировать"
    )

if btn_generate:
    print('clicked in the end')
    print('on generate clicked')
    # Get num_rooms and difficulty, create label
    room_difficulty_int = 0
    if room_difficulty == 'Средний':
        room_difficulty_int = 1
    elif room_difficulty == 'Тяжёлый':
        room_difficulty_int = 2
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMAGE_SIZE, GEN_EMBEDDING).to(device)
    critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC, NUM_CLASSES, IMAGE_SIZE).to(device)
    load_checkpoint(torch.load("C:/MAI/DIPL/BQTMapGeneration/results_18/cgan_map_generation.pth.tar"), gen, critic)
    # gen.eval()
    label_int = (num_rooms - 2) * 3 + room_difficulty_int
    label = torch.LongTensor([
        label_int
    ]).to(device)
    # Pass noise and label in gen
    noise = torch.randn(1, Z_DIM, 1, 1).to(device)
    generated = gen(noise, label)
    # Reformat gen output to numpy or image and add it to app
    generated_img = (generated.squeeze(0)
                     .permute(1, 2, 0)
                     .detach().numpy())  # Into Numpy
    generated_img = generated_img.squeeze(2)
    print(generated_img)
    img = Image.fromarray((generated_img * 255).astype(np.uint8), "L")
    img = to3colors(img)
    img = cleanBorders(img)
    img = cleanSingleGrayPixels(img)
    img = cleanSingleWhitePixels(img)
    img = fillBlackInRooms(img)
    img = fillBlackInRooms(img)
    img = replaceWhiteEntities(img, num_rooms, room_difficulty)
    img = img.resize((512, 512), resample=Image.NEAREST)
    st.image(img)
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_img = buf.getvalue()
    btn_save_img = col[3].download_button(
        label="Сохранить изображение",
        data=byte_img,
        file_name="generated_map.png",
        mime="image/png",
    )



