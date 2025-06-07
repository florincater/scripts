import streamlit as st
import gymnasium as gym
import numpy as np
import torch
from PIL import Image

# Charger le modèle ( modèle entraîné)
@st.cache_resource
def load_model():
    model = torch.load("C:/Users/flori/Lunatic/models/lunar_lander.pth")  # Adaptez au format du modèle
    return model

model = load_model()
# Initialiser l'environnement
env = gym.make("LunarLander-v3", render_mode="rgb_array")
obs, _ = env.reset()

# Interface Streamlit
st.title("🌔 Lunar Lander - Prototype RL")
start_button = st.button("Démarrer la simulation")
if start_button:
    frames = []
    for _ in range(500):  # 500 étapes max
        action = model(torch.FloatTensor(obs)).argmax().item()  # Inference
        obs, _, done, _, _ = env.step(action)
        frame = env.render()
        frames.append(Image.fromarray(frame))  # Convertir en image PIL
        if done:
            obs, _ = env.reset()
            break
    #st.write("Simulation terminée.")

    # Afficher les images
     # Afficher l'animation
    st.image(frames, width=300, caption="Simulation en cours...")

st.write("Appuyez sur le bouton pour lancer l'atterrissage !")
