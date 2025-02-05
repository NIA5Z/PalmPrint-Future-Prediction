from PalmistryAI import PalmistryAI

n_threads is a how many thread PalmistryAI is going to use in your CPU
GPT Model is the large language model we are using.


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
