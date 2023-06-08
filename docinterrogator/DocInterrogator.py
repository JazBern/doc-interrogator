import re
import os
import torch
from PIL import Image
from pdf2image import convert_from_path
from transformers import Pix2StructProcessor as psp
from transformers import Pix2StructForConditionalGeneration as psg
from transformers import pipeline, DonutProcessor, VisionEncoderDecoderModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DocInterrogator:

    def __init__(self, model_type = 'layoutlm') -> None:
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_model(self.model_type)

    def _init_model(self,model_type) -> None:
        if self.model_type == 'donut':
            self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
            self.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        if self.model_type == 'pix2struct':
            self.model = psg.from_pretrained("google/pix2struct-docvqa-large").to(self.device)
            self.processor = psp.from_pretrained("google/pix2struct-docvqa-large")

    def _convert_pdf_to_image(self,pdf_path,page_no) -> Image:
        return convert_from_path(pdf_path,first_page=page_no,last_page=page_no)[0]
    
    def extract_answers(self, image_path, questions: list, page_no=1) -> list:
        model_mapping = {
            'donut': self._extract_donut_answers,
            'layoutlm': self._extract_layoutlm_answers,
            'pix2struct': self._extract_pix2struct_answers
        }

        if image_path.endswith(('.png', '.jpg')):
            image = Image.open(image_path)
        elif image_path.endswith('.pdf'):
            image = self._convert_pdf_to_image(image_path, page_no)

        if extraction_func := model_mapping.get(self.model_type):
            return extraction_func(image, questions)
    
    def _extract_donut_answers(self,image,questions: list) -> list:
        answers = []
        self.model.to(self.device)
        task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
        for question in questions:
            prompt = task_prompt.replace("{user_input}", question)
            decoder_input_ids = self.processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

            pixel_values = self.processor(image, return_tensors="pt").pixel_values

            outputs = self.model.generate(
                pixel_values.to(self.device),
                decoder_input_ids=decoder_input_ids.to(self.device),
                max_length=self.model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
            answers.append(self.processor.token2json(sequence))
        return answers

    def _extract_layoutlm_answers(self,image,questions: list) -> list:
        answers = []
        for question in questions:
            nlp = pipeline(
                    "document-question-answering",
                    model="impira/layoutlm-document-qa",
            )

            answer = nlp( 
                image,
                question
            )[0]['answer']
            answers.append(answer)
        return answers

    def _extract_pix2struct_answers(self,image,questions: list) -> list:
        answers = []
        inputs = self.processor(images=[image for _ in range(len(questions))],text=questions, return_tensors="pt").to(self.device)
        predictions = self.model.generate(**inputs, max_new_tokens=256)
        outputs = self.processor.batch_decode(predictions, skip_special_tokens=True)
        return list(outputs)
