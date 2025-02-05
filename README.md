# Basic
This is a fun project so I will not be providing any support for it and yes this is one off thing.

n_threads is a how many thread PalmistryAI is going to use in your CPU
GPT Model is the large language model we are using.

I use yolo11s image segmentation trained on 33 image so it might not be the best.

# How to use
```python
from PalmistryAI import PalmistryAI

# Initialize the PalmistryAI system
palm_reader = PalmistryAI(
    model_path='./bin/model/Y11SegS.pt',  # Adjust the path if necessary
    gpt_model="Llama-3.2-3B-Instruct-Q4_0.gguf",  
    n_threads=8 
)

# Path to the palm image
image_path = "./EX.png"  # Replace with your image path

# Perform palmistry analysis
try:
    prediction = palm_reader.analyze_palmprint(image_path, verbose=True)
    print("\nPalmistry Reading:\n", prediction)
except FileNotFoundError as e:
    print(e)
```
