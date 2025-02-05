import torch
import cv2
from ultralytics import YOLO
from gpt4all import GPT4All

class PalmistryAI:
    def __init__(self, model_path='./bin/model/Y11SegS.pt', gpt_model="Llama-3.2-3B-Instruct-Q4_0.gguf", n_threads=8):
        self.device = "gpu" if torch.cuda.is_available() else "cpu"
        self.detection_model = YOLO(model_path)
        self.gpt = GPT4All(gpt_model, allow_download=True, n_threads=n_threads, device=self.device)
        self.system_prompt = self._generate_system_prompt()
    
    def _generate_system_prompt(self):
        return """
        You are an expert palm reader. Given a set of palmprint characteristics, your task is to interpret them and predict the user's future with insight and clarity.

        ### Instructions:
        - Analyze the provided palmprint details.
        - Use traditional palmistry knowledge to explain the significance of each print.
        - Offer a balanced prediction, including aspects of love, career, health, and destiny.
        - Be descriptive but concise, avoiding overly vague or generic statements.
        - Keep the tone mystical yet logical.

        ### Palmprint Guide:
        1. **Heart Line** – Represents emotions, love life, and relationships.
        2. **Head Line** – Indicates intellect, thought process, and decision-making ability.
        3. **Life Line** – Reflects vitality, health, and major life events.
        4. **Fate Line** – Reveals career path, achievements, and external influences.
        5. **Sun Line** – Shows fame, success, and personal fulfillment.

        Use these insights to generate a unique and engaging palmistry reading.
        """
    
    def analyze_palmprint(self, image_path, verbose=False):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Sorry, I cannot find {image_path}")
        
        image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
        results = self.detection_model.predict(image, conf=0.65,verbose=verbose)
        
        result_list = []
        for result in results:
            detected = result.boxes.cls.cpu().numpy()
            class_names = self.detection_model.names
            result_list = [class_names[int(key)] for key in detected]
        
        palmprint_text = "I have " + ', '.join(result_list)
        
        prompt = f"""
        Use the given context to predict the user's future based on their palmprint.

        Context:
        - Print 1 represents the Heart Line.
        - Print 2 represents the Head Line.
        - Print 3 represents the Life Line.
        - Print 4 represents the Fate Line.
        - Print 5 represents the Sun Line.
        palmprint: {palmprint_text}
        """
        
        with self.gpt.chat_session(system_prompt=self.system_prompt):
            result_text = self.gpt.generate(prompt, max_tokens=512)
        
        return result_text
