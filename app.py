import streamlit as st
from fastai.vision.all import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# title
st.title('Olma, Lemon, Mango mevalarini klassifikasiya qiluvchi model')

# rasmni joylash
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg'])
#if fale:
  #st.image(file)
  # PIL convert
  img = PILImage.create(file)

  # model
  model = load_learner('fruit_model.pkl')

  # prediction
  pred, pred_id, probs = model.predict(img)
  st.success(f"Bashorat: {pred}")
  st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}')

  # plotting
  #fig = px.bar(x=probs*100, y=model.vocab)
  #st.plotly_chart(fig)
