import os
import pytest
from docinterrogator.DocInterrogator import DocInterrogator

current_dir = os.path.dirname(os.path.abspath(__file__))
sample_pdf = os.path.join(current_dir, "sample.pdf")


def test_interrogator():
    questions = ["What is the name of the company?", "What is the name of the employee?"]
    di = DocInterrogator(model_type="layoutlm")
    answers = di.extract_answers(sample_pdf, questions)
    assert answers[1] == "Sally Harley"