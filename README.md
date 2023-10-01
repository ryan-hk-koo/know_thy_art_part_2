# Know Thy Art Part Two
- This project serves as both a continuation and an extension of [Know_Thy_Art_Part_1](https://github.com/ryan-hk-koo/know_thy_art_part_1)  
- As a continuation, an audio component was added to 'Know Thy Art' in identifying the style of the input painting and highlighting the nuances of each art style (Western Art Style Insight Tool)
- As an extension, new features, namely 'Speech to Art to Speech' and the 'Art Chatbot,' were developed and incorporated into the service 

<br>

- Streamlit Demo on Youtube (turn sound on!)

[![Streamlit Demo](https://img.youtube.com/vi/CgBtw9AcVYY/0.jpg)](https://youtu.be/CgBtw9AcVYY)

<br>

# Purpose
Building on 'Know Thy Art Part 1' — which aimed to transform and deepen art engagement — our second phase prioritizes art accessibility and interaction, especially for those with visual impairments. 'Know Thy Art Part 2' maintains our foundational philosophy of deepening connections with art, while introducing the following new innovative tools: 

- **Know Thy Art Audio Assistance**: Complementing our original work, we integrated an audio component to make art more accessible to visually impaired individuals, allowing them to engage with the various styles of the paintings and their nuances 
- **Speech To Art to Speech**: Inspired by Text-to-Image and Image-to-Text AI models, this feature lets users generate art through speech and then translate visual art back into auditory descriptions. This empowers visually impaired individuals to both create and understand art in a new dimension
- **Art Chatbot**: A chatbot that allows users ask specific questions about art, bridging any information gaps not covered by our other tools

<br>

# Know Thy Art Audio Assistance
![image](https://github.com/ryan-hk-koo/know_thy_art_part_2/assets/143580734/97a2b0ee-5ed9-45ab-b117-6ca19b8318f6)
- Used the Naver CLOVA Voice API to convert text into speech due to its high-quality, natural-sounding voice outputs and its customization options

<br>

# Speech to Art to Speech
![image](https://github.com/ryan-hk-koo/know_thy_art_part_2/assets/143580734/2c7a6d5f-60ac-41e4-8fea-1b8cc1f42e94)
- Utilized a combination of above four tools and Naver CLOVA Voice API to develop the 'Speech to Art to Speech' service
  - **Google Web Speech**: Selected for speech-to-text conversion due to its high accuracy in transcribing spoken words and its capability to filter out background noise
  - **DeepL**: Selected for translating between Korean and English due to its contextual understanding and high-quality translations powered by deep learning models
  - **Stable Diffusion XL**: Selected for text-to-image generation. Being open-source, we were able to train it with our dataset, achieving much better results specifically for impressionism and surrealism styles
  - **mPLUG-Owl**: Selected for image & prompt-to-text generation due to its high degree of accuracy. It ranks the highest among the multi-modal LLMs we considered, as evidenced by this [leaderboard](https://opencompass.org.cn/leaderboard-multimodal)
    - The other multi-modal LLMs considered were MiniGPT-4 and InstructBLIP

<br>

## Fine-Tuning Stable Diffusion XL

![image](https://github.com/ryan-hk-koo/know_thy_art_part_2/assets/143580734/b87ab8b5-b5ff-446c-90ff-a3ff7f5dfb44)

<br>

### Fine-Tuning for Impressionism Style: Before & After 
![image](https://github.com/ryan-hk-koo/know_thy_art_part_2/assets/143580734/a5960fdf-150d-4536-8007-52098ac06e3c)


<br>

### Fine-Tuning for Surrealism Style: Before & After
![image](https://github.com/ryan-hk-koo/know_thy_art_part_2/assets/143580734/b0192dc7-0bd0-4a3e-9075-343175376481)


<br>

# Art Chatbot
- Designed to address user questions about art, deriving answers from a mix of database queries and web searches
- Employs OpenAI's GPT models (text-davinci-003 & gpt-3.5-turbo-16k) with prompt engineering to determine if a user's question is art-related
- Upon identifying art-centric queries, the chatbot taps into the database for answers. This process involves using prompt engineering with the GPT models to craft SQLite3 query statements
- If the database doesn't yield a match, the service, through prompt engineering, formulates a Google search term. Subsequently, the chatbot employs Selenium with a headless ChromeDriver, aided by BeautifulSoup, to fetch detailed answers from the web
- Users are encouraged to keep their inquiries art-specific if they deviate from the topic

<br>

# Conclusion & Reflections
- The 'Speech to Art to Speech' service was successfully developed by integrating multiple tools. However, this integration led to slower processing time for the service
- Fine-tuning the Stable Diffusion XL with our dataset significantly enhanced the quality of the image outputs, especially for the impressionism and surrealism styles
- The chatbot, while efficient, has some reliability concerns due to its partial dependency on internet searches. This underscores the need for a more extensive in-house database to boost its question-answering capabilities
- Both 'Know Thy Art Part One' and 'Know Thy Art Part Two' demonstrate the exciting possibilities when art meets artificial intelligence. In this domain, the potential is vast, opening doors to boundless opportunities


