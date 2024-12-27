Video AI Tools

# ASR Service
Use openai/whisper-large-v3 [hf link here](https://huggingface.co/openai/whisper-large-v3)


## Installation
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
    


