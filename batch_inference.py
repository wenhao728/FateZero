#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/05/15 17:29:21
@Desc    :   Cleaned up batch inference template
@Ref     :   
'''
import os
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel

from save_video import save_video
from test_fatezero import collate_fn
from video_diffusion.models.unet_3d_condition import UNetPseudo3DConditionModel
from video_diffusion.data.dataset import ImageSequenceDataset
from video_diffusion.common.image_util import save_gif_mp4_folder_type
from video_diffusion.common.instantiate_from_config import instantiate_from_config


POS_PROMPT = (
    " ,best quality, extremely detailed, HD, ultra, 8K, HQ, masterpiece, trending on artstation, art, smooth")
NEG_PROMPT = (
    "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, "
    "low quality, deformed body, bloated, ugly, blurry, low res, unaesthetic"
)

data_root = '/data/trc/videdit-benchmark/DynEdit'
method_name = 'fatezero'

config = OmegaConf.create(dict(
    data_root=data_root,
    config_file=f'{data_root}/config.yaml',
    output_dir=f'{data_root}/outputs/{method_name}',
    seed=33,
    # TODO define arguments
    pretrained_model_path='/data/trc/tmp-swh/models/stable-diffusion-v1-5',
    gradient_accumulation_steps=1,
    mixed_precision="fp16",
    test_pipeline_config={
        'target': 'video_diffusion.pipelines.p2p_ddim_spatial_temporal.P2pDDIMSpatioTemporalPipeline',
        'num_inference_steps': 50,
    },
    num_inference_steps=50,
    batch_size=1,
    n_sample_frame=24,
    sampling_rate=1,  # subsample frames
    stride=1,
    guidance_scale=7.5,
    fps=12,
))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # load model
    print('Loading models ...')
    device = torch.device('cuda')
    # TODO define model
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(
        config.pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_path, subfolder="vae",)
    unet = UNetPseudo3DConditionModel.from_2d_model(
        os.path.join(config.pretrained_model_path, "unet"), 
        model_config=dict(lora=160, SparseCausalAttention_index=['mid'], least_sc_channel=640)
    )
    pipeline = instantiate_from_config(
        config.test_pipeline_config,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(config.pretrained_model_path, subfolder="scheduler"),
        disk_store=False,
    )
    pipeline.scheduler.set_timesteps(config.num_inference_steps)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_xformers_memory_efficient_attention()

    data_config = OmegaConf.load(config.config_file)
    set_seed(config.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    text_encoder.eval()
    unet.eval()

    preprocess_elapsed_ls = []
    inference_elapsed_ls = []
    for row in tqdm(data_config['data']):
        output_dir = Path(f"{config.output_dir}/{row.video_id}")
        if output_dir.exists():
            print(f"Skip {row.video_id} ...")
            continue
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        # load video
        print(f"Processing {row.video_id} ...")
        video_path = f'{config.data_root}/frames/{row.video_id}'
        # TODO load video
        prompt_ids = tokenizer(
            row["prompt"], 
            truncation=True, padding="max_length", 
            max_length=tokenizer.model_max_length, return_tensors="pt",
        ).input_ids
        video_dataset = ImageSequenceDataset(
            path=video_path, 
            prompt_ids=prompt_ids,
            prompt=row['prompt'],
            start_sample_frame=0,
            n_sample_frame=config.n_sample_frame,
            sampling_rate=config.sampling_rate,
            stride=config.stride,
        )
        train_dataloader = torch.utils.data.DataLoader(
            video_dataset,
            batch_size=config.batch_size, shuffle=True, num_workers=4,
            collate_fn=collate_fn,
        )
        unet, train_dataloader = accelerator.prepare(unet, train_dataloader)
        train_data_yielder = make_data_yielder(train_dataloader)
        batch = next(train_data_yielder)
        assert batch["images"].shape[0] == 1, "Only support, overfiting on a single video"

        # # Optional
        # inverse_path = Path(f"{config.output_dir}/{row.video_id}/.cache")
        # inverse_path.mkdir(parents=True, exist_ok=True)
        
        # preprocess
        start = time.perf_counter()
        # TODO preprocess video
        text_embeddings = pipeline._encode_prompt(
            row.prompt,
            device=accelerator.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None,
        )
        batch['latents_all_step'] = pipeline.prepare_latents_ddim_inverted(
            rearrange(batch["images"].to(dtype=weight_dtype), "b c f h w -> (b f) c h w"),
            batch_size=1,
            num_images_per_prompt=1,  # not sure how to use it
            text_embeddings=text_embeddings,
            prompt=row.prompt,
            store_attention=True,
            LOW_RESOURCE=True, # not classifier-free guidance
            save_path=None,
        )
        batch['ddim_init_latents'] = batch['latents_all_step'][-1]
        vae.eval()
        text_encoder.eval()
        unet.eval()
        # Convert images to latent space
        images = batch["images"].to(dtype=weight_dtype)
        images = rearrange(images, "b c f h w -> (b f) c h w")
        unet.eval()
        torch.cuda.empty_cache()

        preprocess_elapsed = time.perf_counter() - start
        preprocess_elapsed_ls.append(preprocess_elapsed)

        # edit
        print(f'Editting {row.video_id} ...')
        start = time.perf_counter()
        for i, edit in tqdm(enumerate(row.edit)):
            # TODO edit
            # prompts=edit['prompt'],
            # negative_prompts=edit['src_words']+negative_prompt,
            # inversion_prompt=row['prompt'],
            # edit['tgt_words']
            if edit['type'].startswith('compound'):
                edit['type'] = edit['type'].replace('compound:', '')
            blend_words = []
            equilizer_params={'words':[], 'values':[]}
            for edit_type, src, tgt in zip(
                edit['type'].split(','), edit['src_words'].split(','), edit['tgt_words'].split(',')
            ):
                if edit_type == 'stylization':
                    continue
                elif edit_type == 'foreground' or edit_type == 'background':
                    if len(blend_words) == 0:
                        blend_words.append([src,])
                        blend_words.append([tgt,])
                    else:
                        blend_words[0].append(src)
                        blend_words[1].append(tgt)
                else:
                    raise ValueError(f"Unknown edit type {edit_type}")
                equilizer_params['words'].append(tgt)
                equilizer_params['values'].append(2)
            if len(blend_words) == 0:
                blend_words = None

            p2p_config = {
                'save_self_attention': False,
                'cross_replace_steps': {'default': 0.8},
                'self_replace_steps': 0.8,
                'blend_words': blend_words,
                'eq_params': equilizer_params,
                'use_inversion_attention': True,
                'blend_self_attention':  True,
                'blend_th': [2, 2],
            }
            sequence_return = pipeline(
                prompt=edit['prompt'],
                source_prompt=row.prompt,
                edit_type='swap',
                image=images, # torch.Size([8, 3, 512, 512])
                strength=None,
                generator=generator,
                num_inference_steps=config.num_inference_steps,
                clip_length=config.n_sample_frame,
                guidance_scale=config.guidance_scale,
                num_images_per_prompt=1,
                # used in null inversion
                latents=batch['ddim_init_latents'],
                uncond_embeddings_list=None,
                save_path=None,
                **p2p_config,
            )
            sequence = sequence_return['sdimage_output'].images[0]
            # attention_output = sequence_return['attention_output']
            # save_gif_mp4_folder_type(sequence, str((output_dir / f"{i}.mp4").stem), save_gif=False)
            save_video(output_dir / f"{i}.mp4", sequence, fps=config.fps)
            torch.cuda.empty_cache()

        inference_elapsed = time.perf_counter() - start
        inference_elapsed_ls.append(inference_elapsed)

    with open(f'{config.output_dir}/time.log', 'a') as f:
        f.write(f'Preprocess: {sum(preprocess_elapsed_ls)/len(preprocess_elapsed_ls):.2f} sec/video\n')
        n_prompts = len(row.edit)
        f.write(f'Edit:       {sum(inference_elapsed_ls)/len(inference_elapsed_ls)/n_prompts:.2f} sec/edit\n')
        f.write('Preprocess:\n')
        f.writelines([f'{e:.1f} ' for e in preprocess_elapsed_ls])
        f.write('\nEdit:\n')
        f.writelines([f'{e:.1f} ' for e in inference_elapsed_ls])
        f.write('\n')
    print('Everything done!')


if __name__ == '__main__':
    main()