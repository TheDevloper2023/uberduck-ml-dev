import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:215"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gradio as gr
from utils import e2eSynthesize, download_models
import torch
torch.cuda.empty_cache()



def get_models(type):
    try:
        models = os.listdir(os.path.join("models", type))
        return models
    except Exception as e:
        return []

def refresh_models(current_taco, current_hifi):
    return [
        gr.Dropdown(choices=get_models("tacotron2"), value=current_taco),
        gr.Dropdown(choices=get_models("hifigan"), value=current_hifi)
    ]


with gr.Blocks() as webui:
    with gr.Tab("Inference"):
        with gr.Row():
            with gr.Column():
                tacotron2_model_list = gr.Dropdown(
                    choices=get_models("tacotron2"), 
                    label="Tacotron2 Models", 
                    interactive=True
                )
                hifigan_model_list = gr.Dropdown(
                    choices=get_models("hifigan"), 
                    label="Hifi-GAN Models", 
                    interactive=True
                )
                refresh = gr.Button("🔄 Refresh Models")
                
            with gr.Column():   
                torchmoji_overwrite = gr.Textbox(
                    label="Torchmoji Overwrite", 
                    lines=1, 
                    max_lines=1, 
                    interactive=True
                )
                input_text = gr.Textbox(
                    label="Text", 
                    lines=3, 
                    max_lines=5, 
                    interactive=True
                )
                speaker_weighing = gr.Textbox(
                    label="Speaker Weighing", 
                    lines=1, 
                    max_lines=1, 
                    interactive=True
                )
                    
        with gr.Row():
            synthise = gr.Button("✨ Synthesize", variant="primary")
        
        gr.Markdown("---")
        with gr.Row():
            with gr.Column():
                audio_out = gr.Audio(label="Output", interactive=False)
                error_output = gr.Textbox(label="Error Messages", interactive=False)
            with gr.Column():
                alignment_plot = gr.Image(label="Alignment and Mel Spectrogram", interactive=False)

    
    with gr.Tab("Settings"):
        gr.Markdown("### Tacotron Settings")
        with gr.Group():
            synth_len = gr.Slider(
                label="Max Synthesis Length (seconds)",
                minimum=1,
                maximum=120,
                value=30,
                step=1,
                info="Maximum audio duration the model will generate",
                interactive=True
            )
            symbol_set = gr.Dropdown(
                label="Phoneme Symbol Set",
                choices=["nvidia_taco2", "ipa", "portuguese", "polish", 
                         "dutch", "spanish", "norwegian", "russian", 
                         "ukrainian", "turkish", "dutch", "romanian"],
                value="nvidia_taco2",
                info="Set language symbol set (use 'nvidia_taco2' for English)",
                interactive=True
            )
            with gr.Row():
                arpabet = gr.Checkbox(
                    label="Enable ARPAbet",
                    value=True,
                    info="Use ARPAbet phoneme representation",
                    interactive=True
                )
            split_text = gr.Dropdown(
                label="Text Segmentation Mode",
                choices=[
                    "Feed entire text into model",
                    "Split text by lines",
                    "Split text by sentences",
                    "Split text by sentences and quotes"
                ],
                value="Split text by sentences and quotes",
                interactive=True,
                info="Idk how to explain."
            )

            textseg_len_target = gr.Slider(
                label="Text Segment Length Target",
                minimum=100,
                maximum=2000,
                value=800,
                step=50,
                info="Target length for text segmentation",
                interactive=True
            )

        gr.Markdown("---\n### HiFi-GAN Settings")
        with gr.Group():
            denoise1 = gr.Slider(
                label="Denoiser Strength (Primary)",
                value=35,
                step=0.1,
                minimum=0,
                maximum=100,
                info="Noise reduction strength for main audio pass",
                interactive=True
            )
            denoise2 = gr.Slider(
                label="Denoiser Strength (SR)",
                value=35,
                step=0.1,
                minimum=0,
                maximum=100,
                info="Secondary noise reduction for sample rate adjustment",
                interactive=True
            )
            superres_strength = gr.Slider(
                label="Intensity of superresolution effect (only used if SR is not skipped)",
                value=4.0,
                maximum=10,
                minimum=0,
                step=0.1,
                info="Controls the strength of the superresolution effect",
                interactive=True
            )

            skip_sr = gr.Checkbox(
                label="Skip Super-Resolution",
                value=False,
                info="If checked, SR will be skipped. Recomended if the output sounds wierd.",
                interactive=True
            )
    with gr.Tab("Download Models"):
        model_type = gr.Dropdown (
            label="Model type",
            choices=["tacotron2","hifigan"],
        )
        model_id = gr.Textbox (
            label="Model's Google Drive ID"
        )

        download_btn = gr.Button("Download", variant="secondary")

    
    download_btn.click(
        fn=download_models,
        inputs=[model_type,model_id],
        outputs=None
    )

    refresh.click(
        fn=refresh_models,
        inputs=[tacotron2_model_list, hifigan_model_list],
        outputs=[tacotron2_model_list, hifigan_model_list]
    )

    synthise.click(
        fn=e2eSynthesize,
                inputs=[tacotron2_model_list, hifigan_model_list, input_text, torchmoji_overwrite,
                speaker_weighing, synth_len, symbol_set, arpabet, denoise1, denoise2,
                superres_strength, split_text, textseg_len_target, skip_sr],
        outputs=[audio_out, alignment_plot]
    )

webui.launch(debug=False, share=False)
