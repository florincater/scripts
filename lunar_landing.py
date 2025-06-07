import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st 
import torch  # Import principal
import numpy as np
import gymnasium as gym
from PIL import Image
from ANN import ANN  # Après avoir corrigé ANN.py
disable_env_checker=True  # Désactive les vérifications problématiques
autoreset=True           # Gère mieux les réinitialisations

@st.cache_resource
def load_model():
    model = ANN(state_size=8, action_size=4)
    # Chargement avec chemin absolu

    model_path = os.path.join(os.path.dirname(__file__), "lunar_lander.pth")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

st.title("🌘 Lunar Lander Demo")
model = load_model()

if st.button("Démarrer la simulation"):
    import gymnasium as gym

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset()
    frames = []
    
    for _ in range(620):
        with torch.no_grad():
            action = model(torch.FloatTensor(obs)).argmax().item()
        obs, _, done, _, _ = env.step(action)
        frames.append(Image.fromarray(env.render()))
        if done:
            break
    
    st.image(frames, width=400)
    env.close()
st.write("Simulation terminée. Appuyez sur le bouton pour redémarrer.")