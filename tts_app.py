import gradio as gr
import os
import logging
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["USER"] = "voicecraft" # TODO change this to your username
import time


import torch
import torchaudio
import numpy as np
import random

from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)

import subprocess

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"

import models.voicecraft as voicecraft

valid_weights = ['gigaHalfLibri330M_TTSEnhanced_max16s', 'giga830M' ,'giga330M']
class VCModel:
    def __init__(self, ap) -> None:
        self.voicecraft_name="giga830M.pth" # or giga330M.pth
        #self.voicecraft_name="gigaHalfLibri330M_TTSEnhanced_max16s.pth"
        #self.voicecraft_name="giga330M.pth" 
        self.ckpt_fn =f"./pretrained_models/{self.voicecraft_name}"
        self.encodec_fn = "./pretrained_models/encodec_4cb2048_giga.th"
        self.download_models()
        self.ap = ap
        #self.load_models()

    def download_models(self):
        if not os.path.exists(self.ckpt_fn):
            os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/{self.voicecraft_name}\?download\=true")
            os.system(f"mv {self.voicecraft_name}\?download\=true ./pretrained_models/{self.voicecraft_name}")
        if not os.path.exists(self.encodec_fn):
            os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th")
            os.system(f"mv encodec_4cb2048_giga.th ./pretrained_models/encodec_4cb2048_giga.th")

    def load_models(self):
        self.ckpt_fn =f"./pretrained_models/{self.voicecraft_name}"
        self.ckpt = torch.load(self.ckpt_fn, map_location="cpu")
        self.model = voicecraft.VoiceCraft(self.ckpt["config"])
        self.model.load_state_dict(self.ckpt["model"])
        self.model.to(device)
        self.model.eval()

        self.phn2num = self.ckpt['phn2num']

        self.text_tokenizer = TextTokenizer(backend="espeak")
        self.audio_tokenizer = AudioTokenizer(signature=self.encodec_fn, device=device) # will also put the neural codec model on gpu

    def copy_audio_to_temp(self, orig_audio):
        filename = orig_audio.split("/")[-1]
        os.system(f"cp {orig_audio} {temp_folder}") 
        return f"{temp_folder}/{filename}"

    def get_audio_info(self, orig_audio):
        filename = orig_audio.split("/")[-1]
        self.audio_filename = f"{temp_folder}/{filename}"
        return torchaudio.info(self.audio_filename)

    def generate_audio(self, model, orig_audio, transcript, tts_transcript, cut_off_sec, sample_batch_size, seed, stop_repetition):
        if model not in valid_weights:
            model = valid_weights[0]
        
        self.voicecraft_name = model + '.pth'
        self.ckpt_fn =f"./pretrained_models/{self.voicecraft_name}"

        self.download_models()
        self.load_models()

        audio_fn = self.copy_audio_to_temp(orig_audio)

        info = self.get_audio_info(audio_fn)
        # cut_off_sec = 3.01 # Make this into an input
        audio_dur = info.num_frames / info.sample_rate

        if cut_off_sec <= 1 or cut_off_sec > audio_dur:
            cut_off_sec = audio_dur
        # assert cut_off_sec < audio_dur, f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"
        prompt_end_frame = int(cut_off_sec * info.sample_rate)
        # hyperparameters for inference
        codec_audio_sr = 16000
        codec_sr = 50
        top_k = 0
        top_p = 0.8
        temperature = 1
        silence_tokens=[1388,1898,131]
        kvcache = 1 # NOTE if OOM, change this to 0, or try the 330M model

        # NOTE adjust the below three arguments if the generaton is not as good
        #stop_repetition = 3 # NOTE if the model generate long silence, reduce the stop_repetition to 3, 2 or even 1
        # sample_batch_size = 4 # NOTE: if the if there are long silence or unnaturally strecthed words, increase sample_batch_size to 5 or higher. What this will do to the model is that the model will run sample_batch_size examples of the same audio, and pick the one that's the shortest. So if the speech rate of the generated is too fast change it to a smaller number.
        #seed = 11 # change seed if you are still unhappy with the result
        seed_everything(seed)

        print(f"stop_repetition: {stop_repetition}, sample_batch_size: {sample_batch_size}, seed: {seed} cut_off_sec: {cut_off_sec}")

        decode_config = {'top_k': top_k, 'top_p': top_p, 'temperature': temperature, 'stop_repetition': stop_repetition, 'kvcache': kvcache, "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr, "silence_tokens": silence_tokens, "sample_batch_size": sample_batch_size}
        from inference_tts_scale import inference_one_sample

        target_transcript = transcript + ' ' + tts_transcript
        concated_audio, gen_audio = inference_one_sample(self.model, self.ckpt["config"], self.phn2num, self.text_tokenizer, self.audio_tokenizer, audio_fn, target_transcript, device, decode_config, prompt_end_frame)
        concated_audio, gen_audio = concated_audio[0].cpu(), gen_audio[0].cpu()
        return (codec_audio_sr, concated_audio.numpy()[0]), (codec_audio_sr, gen_audio.numpy()[0])

class AudioProcessor:
    def __init__(self) -> None:
        os.makedirs(temp_folder, exist_ok=True)
        pass

    def initialize_mfa(self):
        subprocess.run(["mfa", "model", "download", "dictionary", "english_us_arpa"])
        subprocess.run(["mfa", "model", "download", "acoustic", "english_us_arpa"])

    def align_audio(self, orig_audio, transcript):
        # Copy over the audio
        os.system(f"cp {orig_audio} {temp_folder}") 

        filename = os.path.splitext(orig_audio.split("/")[-1])[0]
        with open(f"{temp_folder}/{filename}.txt", "w") as f:
            f.write(transcript)

        print("Starting alignment")
        #Run MFA to get alignment
        align_temp = f"{temp_folder}/mfa_alignments"
        subprocess.run(["mfa", "align", "-v", "--clean", "-j", "1", "--output_format", "csv", temp_folder, "english_us_arpa", "english_us_arpa", align_temp, "--beam 1000", "--retry_beam", "2000"])
        print("Finishing alignment")


def greet(audio_path, transcript):
    return f"Hello my amazing {audio_path}" 


temp_folder = "./demo/temp"

def get_transcript_from_temp(audio_file, fallback_transcript):
    filename = os.path.splitext(audio_file.split("/")[-1])[0]
    transcript_file = f"{temp_folder}/{filename}.txt"
    if os.path.isfile(transcript_file):
        with open(transcript_file, "r") as f:
            return f.read()
    return fallback_transcript


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def generate_audio(model, orig_audio, transcript, target_transcript, cutoff, sample_batch_size, seed, stop_repetition):
    return vc_model.generate_audio(model, orig_audio, transcript, target_transcript, cutoff, sample_batch_size, seed, stop_repetition)


def align_audio(orig_audio, transcript):
    return ap.align_audio(orig_audio, transcript)


def printclicked():
    print("align_button Clicked")


if __name__ == "__main__":
    logging.basicConfig(filename='voicecraft.log', level=logging.INFO)
    ap = AudioProcessor()
    vc_model = VCModel(ap)
    with gr.Blocks() as demo:
        model_dropdown = gr.Dropdown(choices=valid_weights, value='gigaHalfLibri330M_TTSEnhanced_max16s', label="Weights")
        audio_input = gr.Audio(value="./demo/84_121550_000074_000000.wav" ,type="filepath", label="Audio File") 
        transcript_input = gr.Textbox(value="But when I had approached so near to them The common object, which the sense deceives, Lost not by distance any of its marks,",label="Transcript")
        target_transcript_input = gr.Textbox(value=", because it was revealed to be a hologram",label="Target Transcript")

        
        align_button = gr.Button(value="Align Audio")
        align_button.click(fn=align_audio, inputs=[audio_input, transcript_input])

        load_transcript_button = gr.Button(value="Load Files Previous Transcript")
        load_transcript_button.click(fn=get_transcript_from_temp, inputs=[audio_input, transcript_input] ,outputs=[transcript_input])

        cutoff_slider = gr.Slider(minimum=0, maximum=30, step=0.01, value=3.01, label="Cut off slider")
        sample_batch_size = gr.Number(value=4, precision=0, label="Sample Batch Size")
        stop_repetition = gr.Number(value=3, precision=0, label="Stop Repetition")
        seed = gr.Number(value=11, precision=0, label="Seed")

        audio_out1 = gr.Audio(label="Concatenated Audio") 
        audio_out2 = gr.Audio(label="Generated Audio") 

        generate_button = gr.Button(value="Generate")
        generate_button.click(fn=generate_audio, 
                              inputs=[
                                  model_dropdown,
                                  audio_input, 
                                  transcript_input, 
                                  target_transcript_input, 
                                  cutoff_slider, 
                                  sample_batch_size, 
                                  seed, 
                                  stop_repetition
                                  ],
                              outputs=[
                                  audio_out1, 
                                  audio_out2
                                  ],
                              show_progress='full')
       
        # demo = gr.Interface(fn=greet, 
        #                     inputs=[
        #                         gr.Audio(value="./demo/84_121550_000074_000000.wav" ,type="filepath", label="Audio File"), 
        #                         gr.Textbox(value="But when I had approached so near to them The common object, which the sense deceives, Lost not by distance any of its marks,",label="Transcript"),
        #                         ], 
        #                     outputs=[
        #                         "text",
        #                         align_button,
        #                         ],
        #                     )
    demo.launch()
