import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import copy
from calibrate_qbb import qbb_replace, QBBCalibrator, get_wikitext_data, qbb_replace_random, qbb_replace_no_upd
import os

def prepare_model(model_id, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    os.environ['HF_HOME'] = os.path.join(save_path, "hf_cache")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=save_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        cache_dir=save_path, 
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Success to load model!")

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    ).to(device)
    teacher.eval()
    student = copy.deepcopy(teacher)
    student = qbb_replace(student, k=3, verbose=True)
    return teacher, student

def load_model_random(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    ).to(device)
    teacher.eval()
    student = copy.deepcopy(teacher)
    student = qbb_replace_random(student, k=3, verbose=True)
    return teacher, student

def load_model_no_upd(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    ).to(device)
    teacher.eval()
    student = copy.deepcopy(teacher)
    student = qbb_replace_no_upd(student, k=3, verbose=True)
    return teacher, student

"""
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    ).to(device)
    teacher.eval()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    student = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        trust_remote_code=True
    ).to(device)
    student = qbb_replace(student, k=4, verbose=True)
    return teacher, student
"""

def fit(student, teacher, tokenizer, s1=1.0, s2=1.0, **kwargs):
    for p in teacher.parameters():
        p.requires_grad = False
    trainer = QBBCalibrator(student, teacher, tokenizer, s1, s2)
    trainer.calibrate(**kwargs)
    student.eval()
    return student


WIKITEXT_SAMPLES = [
    " The game 's traditional linear structure was expanded with a non-linear mission system. ",
    " The character 's design was inspired by several existing science fiction series including Star Wars. ",
    " The development of the game began in 2011 and lasted for over three years with a large team. ",
    " It features a combat system where players can use various weapons and magical abilities. ",
    " The story follows a group of survivors in a post-apocalyptic world searching for a new home. "
]
"""
def calculate_ppl(model, tokenizer, text_samples, device):
    model.eval()
    total_loss = 0
    total_length = 0
    for text in text_samples:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(1)
            total_length += input_ids.size(1)
    return torch.exp(torch.tensor(total_loss / total_length)).item()
"""

def calculate_ppl(model, tokenizer, data_samples, device):
    model.eval()
    total_loss = 0
    total_length = 0
    
    for sample in data_samples:
        if isinstance(sample, str):
            inputs = tokenizer(sample, return_tensors="pt").to(device)
            input_ids = inputs.input_ids
        else:
            input_ids = sample.to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            curr_len = input_ids.size(1)
            total_loss += loss.item() * curr_len
            total_length += curr_len
    if total_length == 0: return float('inf')
    return torch.exp(torch.tensor(total_loss / total_length)).item()

def main():
    OFFICIAL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_path = "/root/autodl-tmp/tinyllama_local"
    # prepare_model(OFFICIAL_ID, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher, student = load_model(model_path)
    # teacher, student = load_model_random(model_path)
    # teacher, student = load_model_no_upd(model_path)
    params = {
        "epochs": 1,
        "lr": 1e-7,
        "batch_size": 1
    }
    optimized_student = fit(
        student,
        teacher,
        tokenizer,
        s1=0.001,
        s2=0.01,
        **params
    )
    # optimized_student = student
    device = "cuda"
    test_samples = get_wikitext_data(tokenizer, n_samples=128, seq_len=64, split="test")
    ppl_t = calculate_ppl(teacher, tokenizer, test_samples, device)
    ppl_s = calculate_ppl(optimized_student, tokenizer, test_samples, device)
    print(f"FP16 teacher PPL: {ppl_t:.4f}")
    print(f"QBB student PPL: {ppl_s:.4f}")
    print(f"Loss: {((ppl_s - ppl_t) / ppl_t * 100):.2f}%")
    save_path = "./qbb_student_model"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"training has completed! save_path: {save_path}")

if __name__ == "__main__":
    main()