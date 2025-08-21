import torch, json, emoji, gdown
from uberduck_ml_dev.vendor.tfcompat.hparam import HParams
from uberduck_ml_dev.models.tacotron2 import Tacotron2, DEFAULTS
from uberduck_ml_dev.data_loader import prepare_input_sequence
from uberduck_ml_dev.models.torchmoji import TorchMojiInterface
import numpy as np
import resampy
import scipy.signal
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("hifi-gan")
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import Generator
from denoiser import Denoiser
import subprocess
import os
import matplotlib.pylab as plt
import nltk
nltk.download('averaged_perceptron_tagger_eng')
from scipy.io import wavfile
from scipy.signal import butter, lfilter, firwin
import time
import torchaudio

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

def e2eSynthesize(taco2,hifi,input,torchmoji_override, speaker_weighting,synthesis_length, symbol_set, arpabet=False, denoise1=35, denoise2=35, superres_strength=4):
    global h2
    use_gpu = torch.cuda.is_available()
    taco2path = "models/tacotron2"
    hifipath = "models/hifigan"
    tacotron2_pth = f"{taco2path}/{taco2}"
    hifi_pth = f"{hifipath}/{hifi}"

    EMOJIS = ":joy: :unamused: :weary: :sob: :heart_eyes: \
:pensive: :ok_hand: :blush: :heart: :smirk: \
:grin: :notes: :flushed: :100: :sleeping: \
:relieved: :relaxed: :raised_hands: :two_hearts: :expressionless: \
:sweat_smile: :pray: :confused: :kissing_heart: :heartbeat: \
:neutral_face: :information_desk_person: :disappointed: :see_no_evil: :tired_face: \
:v: :sunglasses: :rage: :thumbsup: :cry: \
:sleepy: :yum: :triumph: :hand: :mask: \
:clap: :eyes: :gun: :persevere: :smiling_imp: \
:sweat: :broken_heart: :yellow_heart: :musical_note: :speak_no_evil: \
:wink: :skull: :confounded: :smile: :stuck_out_tongue_winking_eye: \
:angry: :no_good: :muscle: :facepunch: :purple_heart: \
:sparkling_heart: :blue_heart: :grimacing: :sparkles:".split(' ')
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
        torchmoji_downloaded = True
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
        #         encoder_outputs = torch.cat((encoder_outputs,), dim=2)

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
    text_padded, input_lengths = prepare_input_sequence(
        [input], cpu_run=cpu_run, arpabet=arpabet, symbol_set=symbol_set, text_cleaner=text_cleaners
    )

    embedding = None
    if use_torchmoji == True:
        if torchmoji_override != "":
            embedding = compute_gst([torchmoji_override])
        else:
            embedding = compute_gst([input])
        embedding = torch.FloatTensor(embedding)
        emojis = torchmoji.enc2emojis(embedding)[0]
    else:
        embedding = torch.zeros(1, 1, 200)
    if use_gpu:
        embedding = embedding.cuda()

    speakerembedding = [0] * speaker_count
    for i in speaker_weighting.split(" "):
        if i:
            speakerembedding[int(i.split(":")[0])] = float(i.split(":")[1])
        speakerembedding = torch.FloatTensor(speakerembedding)
    if use_gpu:
        speakerembedding = speakerembedding.cuda()

    input_ = [text_padded, input_lengths, speakerembedding, embedding]
    output = taco.inference(self = taco, inputs = input_) # idk why i need to specify but it complained otherwise
    print("Running Hifi-gan")


    def vocode_mel_spectrogram(mel_postnet):
            """Vocode mel spectrogram using HiFi-GAN with denoising."""
            y_g_hat = hifigan(mel_postnet.float())
            audio = y_g_hat.squeeze() 
            audio *= MAX_WAV_VALUE
            audio_denoised = denoiser(audio.view(1, -1), strength=denoise1)[:, 0].detach().cpu().numpy().reshape(-1)
            return audio_denoised

    def resample_audio(audio, original_sr, target_sr):
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()

            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)  # (1, T)

            resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
            audio_resampled = resampler(audio)

            return audio_resampled.squeeze(0).numpy()

    def apply_super_resolution(base_audio):
            """Apply HiFi-GAN super-resolution with high-pass filtering."""
            wave = base_audio.astype(np.float32) / MAX_WAV_VALUE
            wave = torch.FloatTensor(wave).to(device)
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
            sr_hat = hifigan_sr(mel).squeeze() * MAX_WAV_VALUE
            sr_audio = denoiser_sr(sr_hat.view(1, -1), strength=denoise2)[:, 0].detach().cpu().numpy().reshape(-1)
        
            # Apply high-pass filter using firwin to boost high frequencies
            b = firwin(101, cutoff=10500, fs=h2.sampling_rate, pass_zero=False)
            sr_audio = lfilter(b, 1, sr_audio)
            
            # Adjust super-resolution strength for balanced enhancement
            high_freqs = superres_strength * sr_audio
            return high_freqs.astype(np.float32)

    def merge_audio(original, superres, normalize=True):
            """Merge base and super-resolution audio."""
            min_len = min(len(original), len(superres))
            original = original[:min_len].astype(np.float32)
            superres = superres[:min_len].astype(np.float32)

            merged = original + superres

            if normalize:
                peak = np.max(np.abs(merged))
                if peak > 0:
                    merged *= 0.98 / peak

            merged = np.clip(merged, -1, 1)
            final_audio_int16 = (merged * 32767).astype(np.int16)

            return final_audio_int16
        
    audio_denoised = vocode_mel_spectrogram(output[1][:1])
    audio_denoised = resample_audio(audio_denoised, h.sampling_rate, h2.sampling_rate)

    audio2_denoised = apply_super_resolution(audio_denoised)

    audio_final = merge_audio(audio_denoised, audio2_denoised, normalize=True)

    
    plot_data_tuple = (output[1][:1].float().data.cpu().numpy()[0], output[3].float().data.cpu().numpy()[0].T)
    plot_img = plot_data(plot_data_tuple, figsize=(graph_width/100, graph_height/100))
    
    audio_path = f"output/audio/{str(time.time())}.wav"
    wavfile.write(audio_path,h2.sampling_rate , audio_final)

    return audio_path, plot_img


def download_models(model_type, model_id):
    os.chdir(os.path.join("models",model_type))
    subprocess.run(["gdown","--id",model_id])
    os.chdir("../..")
