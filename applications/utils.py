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
import time

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

    text_cleaners = ["english_cleaners"] if symbol_set == "nvidia_taco2" else ["turkish_cleaners"] if symbol_set == "turkish" else ["basic_cleaners"]

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
        try:
            torchmoji_downloaded
        except NameError:
                subprocess.run(["wget", "https://github.com/johnpaulbin/torchMoji/releases/download/files/pytorch_model.bin", "-O", "models/torchmoji/pytorch_model.bin"])
                subprocess.run(["wget", "https://raw.githubusercontent.com/johnpaulbin/torchMoji/master/model/vocabulary.json", "-O", "models/torchmoji/vocabulary.json"])
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
            "gate_threshold": 0.25
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
    hifigan_sr, h2, denoiser_sr = load_hifi("models/sr/Superres_Twilight_33000", "config_32k", device)

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

    y_g_hat = hifigan(output[1][:1].float())
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio_denoised = denoiser(audio.view(1, -1), strength=denoise1)[:, 0]
    audio_denoised = audio_denoised.detach().cpu().numpy().reshape(-1)

    normalize = (MAX_WAV_VALUE / np.max(np.abs(audio_denoised))) ** 0.85
    audio_denoised = audio_denoised * normalize
    wave = resampy.resample(
                        audio_denoised,
                        h.sampling_rate,
                        h2.sampling_rate,
                        filter="kaiser_best",
                        window=scipy.signal.windows.hann,
                        num_zeros=18,
                    )
    wave_out = wave.astype(np.int16)
    wave = wave / MAX_WAV_VALUE
    wave = torch.FloatTensor(wave).to(torch.device(device))
    new_mel = mel_spectrogram(
                        wave.unsqueeze(0),
                        h2.n_fft,
                        h2.num_mels,
                        h2.sampling_rate,
                        h2.hop_size,
                        h2.win_size,
                        h2.fmin,
                        h2.fmax,
    )
    y_g_hat2 = hifigan_sr(new_mel.to(device))
    audio2 = y_g_hat2.squeeze()
    audio2 = audio2 * MAX_WAV_VALUE
    audio2_denoised = denoiser(audio2.view(1, -1), strength=denoise2)[:, 0]
    audio2_denoised = audio2_denoised.detach().cpu().numpy().reshape(-1)
    
    b = scipy.signal.firwin(
                        101, cutoff=10500, fs=h2.sampling_rate, pass_zero=False
    )
    y = scipy.signal.lfilter(b, [1.0], audio2_denoised)
    y *= superres_strength
    y_out = y.astype(np.int16)
    y_padded = np.zeros(wave_out.shape)
    y_padded[: y_out.shape[0]] = y_out
    sr_mix = wave_out + y_padded
    sr_mix = sr_mix / normalize

    plot = plot_data((output[1][:1].float().data.cpu().numpy()[0],
            output[3].float().data.cpu().numpy()[0].T))
    
    plot_data_tuple = (output[1][:1].float().data.cpu().numpy()[0], output[3].float().data.cpu().numpy()[0].T)
    plot_img = plot_data(plot_data_tuple, figsize=(graph_width/100, graph_height/100))

    audio_path = f"output/audio/{str(time.time())}.wav"
    audio_int16 = sr_mix.astype(np.int16)
    wavfile.write(audio_path,h2.sampling_rate , audio_int16)


    return audio_path, plot_img


def download_models(model_type, model_id):
    os.chdir(os.path.join("models",model_type))
    subprocess.run(["gdown","--id",model_id])
    os.chdir("../..")
   
