# Whisper Large V3 Audio and Video Text Extraction API Server Demo
Welcome to the **Whisper Large V3** Speech-to-Text API Demo! This project utilizes OpenAI’s powerful **Whisper Large V3** model to transcribe audio and video files, segmenting the speech and outputting the results in a structured JSON format. With an easy-to-use HTTP service, this demo allows developers to quickly integrate speech recognition into their projects.

## 🚀 Features
- **Speech Segmentation:** The model processes your audio/video files and splits the transcriptions into logical segments, preserving context and speaker turns.
- **JSON Output:** Each segment is represented as a structured JSON object, providing detailed information such as timestamps, transcription, and confidence scores.
- **Supports Audio/Video Files:** You can send both audio and video files to the service for transcription.
- **HTTP Service:** A simple HTTP API for easy integration into your own applications or workflows.

Using openai/whisper-large-v3 [hf link here](https://huggingface.co/openai/whisper-large-v3)


## 🛠️ Setup and Usage
1. [Optional] env 
 It's recommended to use `conda`, which you can download [here](https://docs.anaconda.com/free/miniconda/miniconda-install/).
    ```shell
    conda create -n mediaai python=3.11.11
    conda activate mediaai
    ```

2. Install Dependency
    ```shell
    cd asr
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

3. Run ASR Service
    ```shell
    python launch_asr_service.py 8080
    ```

4. Test ASR API
    ```shell
    curl -X POST 'http://localhost:8080/asr_seg' -F "file=@/Users/alen/Downloads/v0d00fg10000cti76avog65pg6tif4mg.mp4"
    ```
   Example Response
    ```json
    {
      "msg": "success",
      "paragraph": "Wow, I’m seeing no fewer than a few of my own virtual avatars on the screen. It makes me wonder if AI’s thought power has already surpassed human cognition. In this blazing furnace, the FORCE-powered AI is on its way to achieving the potential for autonomous production. Below are a few key breakthroughs in technology: AI can now create 3D models instantly based on simple text input. With just a few words, it generates a highly accurate digital asset. AI’s ability to quickly produce output is impressive.",
      "sentences": [
        {"start_time": 0.0, "end_time": 2.54, "text": "Wow, I’m seeing no fewer than a few of my own virtual avatars."},
        {"start_time": 2.54, "end_time": 5.96, "text": "It makes me wonder if AI’s thought power has already surpassed human cognition."},
        {"start_time": 5.96, "end_time": 8.36, "text": "In this blazing furnace, the FORCE-powered AI is advancing towards autonomous production."},
        {"start_time": 8.36, "end_time": 10.2, "text": "AI is approaching the potential for self-sufficiency."},
        {"start_time": 10.2, "end_time": 12.52, "text": "Below are a few key breakthroughs in technology."},
        {"start_time": 12.52, "end_time": 15.52, "text": "AI can create 3D models instantly from simple descriptions."},
        {"start_time": 15.52, "end_time": 17.44, "text": "Now you just need to enter a few words for it to create an accurate digital asset."},
        {"start_time": 17.44, "end_time": 19.94, "text": "This demonstrates AI’s capability to quickly generate outputs."}
      ]
    }
   ```
    


