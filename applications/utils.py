import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:215"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch, json
torch.cuda.empty_cache()
from uberduck_ml_dev.vendor.tfcompat.hparam import HParams
from uberduck_ml_dev.models.tacotron2 import Tacotron2, DEFAULTS
from uberduck_ml_dev.data_loader import prepare_input_sequence
from uberduck_ml_dev.models.torchmoji import TorchMojiInterface
import numpy as np
import resampy
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("hifi-gan")
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import Generator
from denoiser import Denoiser
import subprocess
import matplotlib.pylab as plt
import unidecode
import nltk
from nltk import sent_tokenize
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
from scipy.io import wavfile
from scipy.signal import lfilter, firwin
import time
import glob
from unidecode import unidecode

from uberduck_ml_dev.text.util import convert_to_arpabet
def ARPAconverter(text: str) -> str:
    temp = convert_to_arpabet(text=text)
    #Cleaning up stuff. Taken from the arpabet transcriptor thigy
    temp = temp.replace("{,}", ",")
    temp = temp.replace("{;}", ";")
    temp = temp.replace("{:}", ":")
    temp = temp.replace("{-}", "-")
    temp = temp.replace("{...}", ".")
    temp = temp.replace("{ ", "{")
    temp = temp.replace(" }", "}")
    temp = temp.replace("{.}", ".")
    temp = temp.replace("{?}", "?")
    temp = temp.replace("{!}", "!")
    temp = temp.replace("} .", "}.")
    temp = temp.replace("} ,", "},")
    temp = temp.replace("} !", "}!")
    temp = temp.replace("} ?", "}?")
    temp = temp.replace("} '", "}'")
    return temp

graph_width = 900
graph_height = 360
def plot_data(data, figsize=(int(graph_width/100), int(graph_height/100))):
            fig, axes = plt.subplots(1, len(data), figsize=figsize)
            for i in range(len(data)):
                axes[i].imshow(data[i], aspect='auto', origin='lower',
                            interpolation='none', cmap='inferno')
            fig.canvas.draw()
            plt.show()


def load_hifi(path, conf_name, dv):
    conf = os.path.join("hifi-gan", conf_name + ".json")
    with open(conf) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    hifigan = Generator(h).to(torch.device(dv))
    state_dict_g = torch.load(path, map_location=torch.device(dv))
    hifigan.load_state_dict(state_dict_g["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    denoiser = Denoiser(hifigan, mode="zeros")
    return hifigan, h, denoiser

def resample_audio(audio, original_sr, target_sr):
            #if isinstance(audio, np.ndarray):
            #    audio = torch.from_numpy(audio).float()

            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()

            #if len(audio.shape) == 1: #Comment more things.
            #    audio = audio.unsqueeze(0)  # (1, T)

            audio = audio.astype(np.float32) #4 good mesure unf

            #resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr) #This is shit
            #audio_resampled = resampler(audio)
            audio_resampled = resampy.resample(audio.astype(np.float32), original_sr, target_sr, filter='kaiser_best', axis=-1) #This sounds better with Luna so Ill keep it womp womp


            #return audio_resampled.squeeze(0).numpy()
            return audio_resampled


def merge_audio(original, superres, normalize=True):
            """Merge base and super-resolution audio."""
            min_len = min(len(original), len(superres))
            original = original[:min_len].astype(np.float32)
            superres = superres[:min_len].astype(np.float32)

            merged = 0.8 * original + 0.5 * superres

            if normalize:
                peak = np.max(np.abs(merged))
                if peak > 0:
                    merged *= 0.98 / peak

            merged = np.clip(merged, -1, 1)
            final_audio_int16 = (merged * 32767).astype(np.int16)

            return final_audio_int16

def merge_audio_files(file_list, output_path, sampling_rate):
    """Merge multiple audio files into one using scipy."""
    audio_data = []
    for file_path in file_list:
        rate, data = wavfile.read(file_path)
        if rate != sampling_rate:
            # Resample if needed
            data = resample_audio(data, rate, sampling_rate)
        audio_data.append(data)
    
    # Concatenate all audio data
    merged_audio = np.concatenate(audio_data)
    
    # Write the merged file
    wavfile.write(output_path, sampling_rate, merged_audio)
    return output_path

def cleanup_files(file_pattern):
    """Clean up temporary files matching a pattern."""
    for file_path in glob.glob(file_pattern):
        try:
            os.remove(file_path)
        except:
            pass

#Thanks Cookie, uwu

def parse_text_into_segments(texts, split_at_quotes=True, target_segment_length=200, split_at_newline=True):
    """Swap speaker at every quote mark. Each split segment will have quotes around it (for information later on rather than accuracy to the original text)."""
    
    # split text by quotes
    quo ='"' # nested quotes in list comprehension are hard to work with
    wsp =' '
    texts = [f'"{text.replace(quo,"").strip(wsp)}"' if i%2 else text.replace(quo,"").strip(wsp) for i, text in enumerate(unidecode(texts).split('"'))]
    
    # clean up and remove empty texts
    def clean_text(text):
        text = (text.strip(" ")
                   #.replace("\n"," ")
                    .replace("  "," ")
                    .replace("> --------------------------------------------------------------------------","")
                    .replace("------------------------------------",""))
        return text
    texts = [clean_text(text) for text in texts if len(text.replace('"','').strip(' ')) or len(clean_text(text))]
    assert len(texts)
    
    # split text by sentences and add commas in where needed.
    def quotify(seg, text):
        if '"' in text:
            if seg[0] != '"': seg='"'+seg
            if seg[-1] != '"': seg+='"'
        return seg
    texts_tmp = []
    for text in texts:
        if len(text.strip()):
            for x in sent_tokenize(text):
                if len(x.replace('"','').strip(' ')):
                    texts_tmp.extend([quotify(x.strip(" "), text)])
        else:
            if len(text.replace('"','').strip(' ')):
                texts_tmp.extend([quotify(text.strip(" "), text)])
    #texts = [texts_tmp.extend([quotify(x.strip(" "), text) for x in sent_tokenize(text) if len(x.replace('"','').strip(' '))]) for text in texts]
    texts = texts_tmp
    del texts_tmp
    assert len(texts)
    
    # merge neighbouring sentences
    quote_mode = False
    texts_output = []
    texts_segmented = ''
    texts_len = len(texts)
    for i, text in enumerate(texts):
        
        # split segment if quote swap
        if split_at_quotes and ('"' in text and quote_mode == False) or (not '"' in text and quote_mode == True):
            texts_output.append(texts_segmented.replace('"','').replace("\n","").strip())
            texts_segmented=text
            quote_mode = not quote_mode
        # if the prev text is already longer than the target length
        elif len(texts_segmented) > target_segment_length:
            text = text.replace('"','')
            while len(texts_segmented):
                texts_segmented_parts = texts_segmented.split(',')
                texts_segmented = ''
                texts_segmented_overflow = ''
                for i, part in enumerate(texts_segmented_parts):
                    
                    if split_at_newline and part.strip(' ').endswith('\n'):
                        part = part.replace('\n','')
                        texts_segmented += part if i == 0 else f',{part}'
                        texts_segmented_overflow = ','.join(texts_segmented_parts[i+1:])
                        break
                    elif split_at_newline and part.strip(' ').startswith('\n'):
                        part = part.replace('\n','')
                        texts_segmented = ''
                        texts_segmented_overflow = ','.join(texts_segmented_parts[i+1:])
                        break
                    elif i > 0 and len(texts_segmented)+len(part.replace('\n','')) > target_segment_length:
                        texts_segmented_overflow = ','.join(texts_segmented_parts[i:])
                        break
                    else:
                        part = part.replace('\n','')
                        texts_segmented += part if i == 0 else f',{part}'
                
                if len(texts_segmented.replace('\n','').strip()):
                    texts_output.append(texts_segmented.replace('\n','').strip())
                    texts_segmented = ''
                if len(texts_segmented_overflow):
                    texts_segmented = texts_segmented_overflow
                del texts_segmented_overflow, texts_segmented_parts
            texts_segmented=text
        
        else: # continue adding to segment
            text = text.replace('"','')
            texts_segmented+= f' {text}'
    
    # add any remaining stuff.
    while len(texts_segmented):
        texts_segmented_parts = texts_segmented.split(',')
        texts_segmented = ''
        texts_segmented_overflow = ''
        for i, part in enumerate(texts_segmented_parts):
            if i > 0 and len(texts_segmented)+len(part) > target_segment_length:
                texts_segmented_overflow = ','.join(texts_segmented_parts[i:])
                break
            else:
                texts_segmented += part if i == 0 else f',{part}'
        if len(texts_segmented.strip()):
            texts_output.append(texts_segmented.strip())
        texts_segmented = ''
        if len(texts_segmented_overflow):
            texts_segmented = texts_segmented_overflow
        del texts_segmented_overflow, texts_segmented_parts
    
    assert len(texts_output)
    
    return texts_output

def e2eSynthesize(taco2,hifi,input,torchmoji_override, speaker_weighting,synthesis_length, symbol_set, arpabet=True, denoise1=35, denoise2=35, superres_strength=4, textsegmode="Feed entire text into model", textseg_len_target=800, skip_sr=False):

    def apply_super_resolution(base_audio):
        torch.cuda.empty_cache()
        """Apply HiFi-GAN super-resolution with smooth high-frequency enhancement."""
        wave = base_audio.astype(np.float32) / MAX_WAV_VALUE
        wave = torch.FloatTensor(wave).to(device)

        # Mel spectrogram
        mel = mel_spectrogram(
            wave.unsqueeze(0),
            h2.n_fft,
            h2.num_mels,
            h2.sampling_rate,
            h2.hop_size,
            h2.win_size,
            h2.fmin,
            h2.fmax,
        )

        # Run HiFi-GAN SR
        sr_hat = hifigan_sr(mel).squeeze() * MAX_WAV_VALUE
        sr_audio = denoiser_sr(sr_hat.view(1, -1), strength=denoise2)[:, 0].detach().cpu().numpy().reshape(-1)

        # Smooth low-pass to isolate highs (not remove mids!)
        b = firwin(601, cutoff=9000, fs=h2.sampling_rate, pass_zero=False, window=('kaiser', 1.0)).astype(np.float32)
        sr_audio = lfilter(b, 1.0, sr_audio) * 1.5

        # Adjust super-resolution strength for balance
        high_freqs = superres_strength * sr_audio
        return high_freqs.astype(np.float32)
    
    def vocode_mel_spectrogram(mel_postnet):
            """Vocode mel spectrogram using HiFi-GAN with denoising."""
            torch.cuda.empty_cache()
            y_g_hat = hifigan(mel_postnet.float())
            audio = y_g_hat.squeeze().cpu()
            audio *= MAX_WAV_VALUE
            audio_denoised = denoiser(audio.view(1, -1), strength=denoise1)[:, 0].detach().cpu().numpy().reshape(-1)
            return audio_denoised

    global h2
    use_gpu = torch.cuda.is_available()
    taco2path = "models/tacotron2"
    hifipath = "models/hifigan"
    tacotron2_pth = f"{taco2path}/{taco2}"
    hifi_pth = f"{hifipath}/{hifi}"
    config = DEFAULTS.values()

    text_cleaners = ["english_cleaners"] if symbol_set == "nvidia_taco2" else ["turkish_cleaners"] if symbol_set == "turkish" else ["romanian_cleaners"] if symbol_set == "romanian" else ["basic_cleaners"]

    m = torch.load(tacotron2_pth, map_location = torch.device("cpu"))

    if "model" in m.keys():
        m = m["model"]

    if "state_dict" in m.keys():
        m = m["state_dict"]

    if "speaker_embedding.weight" in m.keys():
        speaker_count = len(m["speaker_embedding.weight"])
        config.update(
            {
                "has_speaker_embedding": True,
                "n_speakers": speaker_count,
                "ignore_layers": ["null"],
            }
        )
        print("%s speakers found in model" % speaker_count)
    else:
        print("single-speaker model")
        speaker_count = 1

    if "gst_lin.weight" in m.keys():
        use_torchmoji = True
        config.update(
            {
            "gst_dim": 2304,
            "gst_type": "torchmoji",
            "torchmoji_vocabulary_file": "models/torchmoji/vocabulary.json",
            "torchmoji_model_file": "models/torchmoji/pytorch_model.bin",
            }
        )
        torchmoji = TorchMojiInterface(
            "models/torchmoji/vocabulary.json",
            "models/torchmoji/pytorch_model.bin",
        )
        compute_gst = lambda texts: torchmoji.encode_texts(texts)
    else:
        use_torchmoji = False

    del m

    config.update(
            {
            "max_decoder_steps": synthesis_length * 100,
            "symbol_set": symbol_set,
            "text_cleaners": text_cleaners,
            }
    )

    hparams = HParams(**config)

    taco = Tacotron2(hparams)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    taco.from_pretrained(tacotron2_pth, device=device)

    @torch.no_grad()
    def inference(self, inputs):
        text, input_lengths, speaker_ids, embedded_gst, *_ = inputs

        embedded_inputs = self.embedding(text).transpose(1, 2)
        embedded_text = self.encoder.inference(embedded_inputs, input_lengths)
        encoder_outputs = embedded_text
        if self.speaker_embedding:
            if use_gpu:
                embeddings = self.speaker_embedding(torch.LongTensor([range(self.n_speakers)]).cuda())[0]
            else:
                embeddings = self.speaker_embedding(torch.LongTensor([range(self.n_speakers)]))[0]
            average = torch.mean(embeddings, 0)
            embedding_offsets = torch.stack(tuple(torch.subtract(i, average) for i in embeddings))
            mixed_offset = torch.sum(embedding_offsets * speaker_ids[:, None], 0)
            embedded_speakers = torch.add(mixed_offset, average)
            encoder_outputs += self.spkr_lin(embedded_speakers)

        if self.gst_lin is not None:
            assert (
                embedded_gst is not None
            ), f"embedded_gst is None but gst_type was set to {self.gst_type}"
            encoder_outputs += self.gst_lin(embedded_gst)

        memory_lengths = input_lengths
        mel_outputs, gate_outputs, alignments, mel_lengths = self.decoder.inference(
            encoder_outputs, memory_lengths
        )
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, mel_lengths]
        )

    taco.inference = inference


    hifigan, h, denoiser = load_hifi(hifi_pth, "config_v1", device)
    hifigan_sr, h2, denoiser_sr = load_hifi("models/sr/Superres_Twilight_33000.pth", "config_32k", device)

    cpu_run = not use_gpu

    # Create a unique prefix for this synthesis session
    filename_prefix = f"{int(time.time())}_{hash(input) % 10000:04d}"
    working_dir = "output/temp"
    output_dir = "output/audio"
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if textsegmode == "Feed entire text into model":
         texts = [input,]
    elif textsegmode == "Split text by lines":
         texts = input.split("\n")
    elif textsegmode == "Split text by sentences":
         texts = parse_text_into_segments(input, split_at_quotes=False, target_segment_length=textseg_len_target)
    elif textsegmode == "Split text by sentences and quotes":
         texts = parse_text_into_segments(input, split_at_quotes=True, target_segment_length=textseg_len_target)
    else:
         raise NotImplementedError(f"textseg_mode of {textsegmode} is invalid.")
    
    # Initialize tracking variables
    segment_files = []
    counter = 0
    out_count = 0
    fpaths = []
    running_fsize = 0
    
    total_len = len(texts)
    text_batch_in_progress = []
    continue_from = 0

    # Generate from text splitting
    for text_index, text in enumerate(texts):
        if text_index < continue_from: 
            print(f"Skipping {text_index}.\t", end="")
            continue

        last_text = (text_index == (total_len-1)) # true if final text input
        
        # Setup text batches
        text_batch_in_progress.append(text)
        if (len(text_batch_in_progress) == 1) or last_text: # if text batch ready or final input
            text_batch = text_batch_in_progress
            text_batch_in_progress = []
        else:
            continue # if batch not ready, add another text

        if arpabet == True:
            text_batch = [ARPAconverter(text) for text in text_batch]




        ### Prep inputs
        text_padded, input_lengths = prepare_input_sequence(
            text_batch, cpu_run=cpu_run, arpabet=1.0 if arpabet == True else 0.0, symbol_set=symbol_set, text_cleaner=text_cleaners
        )

        #### Do Torchmoji
        embedding = None
        if use_torchmoji == True:
            if torchmoji_override != "":
                embedding = compute_gst([torchmoji_override])
            else:
                embedding = compute_gst(text_batch)
            embedding = torch.FloatTensor(embedding)
        else:
            embedding = torch.zeros(len(text_batch), 1, 200)
        if use_gpu:
            embedding = embedding.cuda()

        #### Speaker-embeddings
        speakerembedding = [0] * speaker_count
        for i in speaker_weighting.split(" "):
            if i:
                speakerembedding[int(i.split(":")[0])] = float(i.split(":")[1])
        speakerembedding = torch.FloatTensor(speakerembedding)
        if use_gpu:
            speakerembedding = speakerembedding.cuda()
        # Repeat for each item in batch
        speakerembedding = speakerembedding.repeat(len(text_batch), 1)

        #### Generate mel spec
        input_ = [text_padded, input_lengths, speakerembedding, embedding]
        output = taco.inference(self=taco, inputs=input_)[0]

        #### Process mel spec
        # pick first sample in batch
        output_postnet = taco.postnet(output)

        mel_batch_postnet = output + 1.6 *output_postnet

        # Process each item in the batch
        for i in range(len(text_batch)):
            # Vocode the mel spectrogram
            torch.cuda.empty_cache()
            audio_denoised = vocode_mel_spectrogram(mel_batch_postnet)
            sample_rate = None
            if not skip_sr:
                print("Applying super-resolution...")
                audio_denoised = resample_audio(audio_denoised, h.sampling_rate, h2.sampling_rate)
                audio2_denoised = apply_super_resolution(audio_denoised)
                audio_final = merge_audio(audio_denoised, audio2_denoised, normalize=True)
                sample_rate = h2.sampling_rate
            else:
                print("Skipping super-resolution...")
                audio_final = audio_denoised.astype(np.int16)
                sample_rate = h.sampling_rate

            # Save segment to temporary file
            segment_filename = f"{filename_prefix}_{counter:06d}.wav"
            segment_path = os.path.join(working_dir, segment_filename)
            wavfile.write(segment_path, sample_rate, audio_final)
            segment_files.append(segment_path)
            counter += 1

            # Check if we need to merge segments
            if len(segment_files) >= 300 or (last_text and i == len(text_batch)-1 and segment_files):
                # Merge this batch of segments
                merged_filename = f"{filename_prefix}_concat_{out_count:04d}.wav"
                merged_path = os.path.join(working_dir, merged_filename)
                merge_audio_files(segment_files, merged_path, sample_rate)
                
                # Clean up individual segment files
                # Only remove segment files like 000001.wav, 000002.wav, etc.
                cleanup_files(os.path.join(working_dir, f"{filename_prefix}_[0-9][0-9][0-9][0-9][0-9][0-9].wav"))

                
                # Track the merged file for potential further merging
                fsize = os.stat(merged_path).st_size
                running_fsize += fsize
                fpaths.append(merged_path)
                
                # Check if we need to create a final output
                if (running_fsize/(1024**3) > 2.0) or (len(fpaths) > 10) or (last_text and i == len(text_batch)-1):
                    # Create final output file
                    final_output = f"{filename_prefix}_final_{out_count:04d}.wav"
                    final_path = os.path.join(output_dir, final_output)
                    merge_audio_files(fpaths, final_path, sample_rate)
                    
                    # Clean up intermediate merged files
                    cleanup_files(os.path.join(working_dir, f"{filename_prefix}_concat_*.wav"))
                    
                    # Reset tracking variables
                    running_fsize = 0
                    fpaths = []
                    out_count += 1
                
                # Reset segment files for next batch
                segment_files = []

    # Create a plot for the last segment (optional), Ignore for now
    ###plot_data_tuple = (mel_batch_postnet[-1:].float().data.cpu().numpy()[0], output[3][-1:].flcoat().data.cpu().numpy()[0].T)
    ###plot_img = plot_data(plot_data_tuple, figsize=(graph_width/100, graph_height/100))

    
    
    # Return the path to the final audio file
    return final_path, None 


def download_models(model_type, model_id):
    os.chdir(os.path.join("models",model_type))
    subprocess.run(["gdown","--id",model_id])
    os.chdir("../..")
