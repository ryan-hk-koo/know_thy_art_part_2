# Know Thy Art Part Two
- This project serves as both a continuation and an extension of [Know_Thy_Art_Part_1](https://github.com/ryan-hk-koo/know_thy_art_part_1)  
- As a continuation, an audio component was added to 'Know Thy Art' in identifying the style of the input painting and providing a guide on the nuances of each art style (Western Art Style Insight Tool)
- As an extension, new features, namely 'Speech to Art to Speech' and the 'Art Chatbot,' were  developed and incorporated into the service 

<br>

- Streamlit Demo on Youtube (turn sound on!)

[![Streamlit Demo](https://img.youtube.com/vi/CgBtw9AcVYY/0.jpg)](https://youtu.be/CgBtw9AcVYY)

# Purpose
Building upon 'Know Thy Art Part 1' — which sought to deepen art appreciation and engagement — we further envisioned a space where art accessibility and interaction are paramount, especially for those who might experience art differently, such as the visually impaired. The second part of 'Know Thy Art' retains the core philosophy of fostering a profound connection with art but pushes the boundaries by introducing : 

- **Know Thy Art Audio Assistance**: Complementing our original work, we integrated an audio component, ensuring that even visually impaired individuals can engage and resonate with the various styles of the paintings and their nuances 
- **Speech To Art to Speech**: Inspired by Text-to-Image and Image-to-Text AI models, we ventured a step further. Now, users can create art through speech and subsequently translate visual art back into descriptive narratives, empowering blind or visually impaired individuals to both create and comprehend art in an auditory manner
- **Art Chatbot**: Introduced to allow users to pose specific questions about art that are not addressed by our other services, facilitating a deeper exploration of art

With these, we're not only expanding our services, but also strengthening our commitment to making art a personalized, inclusive, and immersive experience for everyone

<br>

# Know Thy Art Audio Assistance
![image](https://github.com/ryan-hk-koo/know_thy_art_part_2/assets/143580734/97a2b0ee-5ed9-45ab-b117-6ca19b8318f6)
- Used the Naver CLOVA Voice API to convert text into speech due to its high-quality, natural-sounding voice outputs and its customization options
<br>

# Speech to Art to Speech
![image](https://github.com/ryan-hk-koo/know_thy_art_part_2/assets/143580734/2c7a6d5f-60ac-41e4-8fea-1b8cc1f42e94)
- Utilized a combination of above four tools and Naver CLOVA Voice API to develop the 'Speech to Art to Speech' service
  - **Google Web Speech**: Chosen for speech-to-text conversion due to its high accuracy in transcribing spoken words and its capability to filter out background noise
  - **DeepL**: Employed for translating between Korean and English, valued for its contextual understanding and high-quality translations powered by deep learning models
  - **Stable Diffusion XL**: Used for text-to-image generation. Being open-source, it allowed us to train with our dataset, achieving better results specifically for impressionism and surrealism styles
  - **mPLUG-Owl**: Selected for image & prompt-to-text generation because of its high degree of accuracy. It ranks the highest among the multi-modal LLMs we considered, as evidenced by this [leaderboard](https://opencompass.org.cn/leaderboard-multimodal)
    - The other multi-modal LLMs considered were MiniGPT-4 and InstructBLIP

<br>

## Training Stable Diffusion XL

![image](https://github.com/ryan-hk-koo/know_thy_art_part_2/assets/143580734/ca0797dc-b2f6-4e57-bcd4-76669f14d18d)

### Impressionism Style Training Before & After 
![image](https://github.com/ryan-hk-koo/know_thy_art_part_2/assets/143580734/ec7d6297-4a16-4895-94be-5a79d94ccb78)

### Surrealism Style Training Before & After 
![image](https://github.com/ryan-hk-koo/know_thy_art_part_2/assets/143580734/e4dd77d5-1577-4008-a3fd-bb950c4e5f18)
