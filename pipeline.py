from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np
import torch
import copy
from torch.nn import functional as F
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.utils import deprecate, is_accelerate_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from PIL import Image
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
import PIL
from diffusers import DDIMScheduler, DDPMScheduler, DDIMInverseScheduler
from diffusers import image_processor
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
import torchvision.transforms as transforms
logger = logging.get_logger(__name__)
from PIL import ImageChops
import time

class MCOWPipeline(StableDiffusionPipeline):

    _optional_components = ["safety_checker", "feature_extractor", "inverse_scheduler"]

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: DDIMScheduler,
            inverse_scheduler: DDIMInverseScheduler,
            feature_extractor: CLIPImageProcessor,
            requires_safety_checker: bool = True,
    ):

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            inverse_scheduler=inverse_scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = image_processor.VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        subprompts: List[str] = None,
        images: List[PIL.Image.Image] = None,
        mask_images: List[PIL.Image.Image] = None,
        image_sizes: List[int] = [256, 256],
        xys: List[List[int]] = [[0, 200], [256, 200]],
        ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
   

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        self.inverse_scheduler_copy = copy.deepcopy(self.inverse_scheduler)

        replace_step = int(0.6 * num_inference_steps)
        back_step = int(0.8 * num_inference_steps)
        replace_to_x0 = [int(0.5 * num_inference_steps)]

        # 3. Preprocess image
        bg_img = Image.new("RGB", (height, width), color=(128, 128, 128)) # default gray
        bg_mask = torch.zeros((1,4,64,64))

        self.mask_latents = []
        self.back2replace = []
        self.replace2V0 = []
        self.xys = xys

        for img, mask, xy, size, sub_prompt in zip(images, mask_images, xys, image_sizes, subprompts):
            x, y = xy

            img = img.resize((size, size))
            mask = mask.resize((size, size))

            mask = mask.convert('1')
            mask = ImageChops.invert(mask)

            img = ImageChops.composite(img, Image.new('RGB', img.size, (128, 128, 128)), mask)
            bg_img.paste(img, (x, y))

            mask = transforms.ToTensor()(mask)
            mask_latent = mask.unsqueeze(0)[:,:1,:,:].repeat(batch_size,4,1,1)
            mask_latent = torch.nn.functional.interpolate(mask_latent, size=(size//self.vae_scale_factor, size//self.vae_scale_factor), mode='nearest')
            mask_latent[mask_latent != 1] = 0
            bg_mask[:,:, y//8:y//8+size//8, x//8:x//8+size//8] = mask_latent
            
            self.mask_latents.append(mask_latent)

            self.inverse_scheduler = copy.deepcopy(self.inverse_scheduler_copy)
            img_tensor = self.image_processor.preprocess(img)
            img_latent = self.prepare_image_latents(img_tensor, batch_size, self.vae.dtype, device, generator)

            V_on_replace, V0_to_Vreplace = self.invert(prompt=sub_prompt, latents=img_latent, num_inference_steps=num_inference_steps,
                                                       start_step=0, end_step=replace_step, save_latens=True,)
            V_on_back, Vreplace_to_Vback = self.invert(prompt=sub_prompt, latents=V_on_replace, num_inference_steps=num_inference_steps, 
                                                       start_step=replace_step, end_step=back_step, save_latens=True,)
            Vreplace_to_V0 = V0_to_Vreplace[::-1]
            Vback_to_Vreplace = Vreplace_to_Vback[::-1]

            self.back2replace.append(Vback_to_Vreplace)
            self.replace2V0.append(Vreplace_to_V0)


        bg_tensor = self.image_processor.preprocess(bg_img)
        bg_latent = self.prepare_image_latents(bg_tensor, batch_size, self.vae.dtype, device, generator)

        self.inverse_scheduler = copy.deepcopy(self.inverse_scheduler_copy)
        bg_on_replace = self.invert(prompt=prompt, latents=bg_latent, num_inference_steps=num_inference_steps,
                                    start_step=0, end_step=replace_step,)

        back_to_replace = [i for i in range(back_step, replace_step-1, -1)]

        for cycle_id in range(10):
            bg_on_back = self.inject_noise(x_t=bg_on_replace, start_step=replace_step, end_step=back_step)
            # back -> replace, and replace in between run
            bg_on_replace = self.sample_and_replace_with_mask(x=bg_on_back, last_step=back_step, end_step=replace_step, back_step=back_step,
                                         replace_step_list=back_to_replace, eta=1., prompt=prompt, generator=generator,
                                         Vs_back_to_front=self.back2replace, mask_latents=self.mask_latents, bg_mask=bg_mask, num_inference_steps=num_inference_steps)

        x0 = self.sample_and_replace(x=bg_on_replace, last_step=replace_step, end_step=0, back_step=replace_step,
                                     replace_step_list=replace_to_x0, eta=0.1, prompt=prompt, generator=generator,
                                     Vs_back_to_front=self.replace2V0, mask_latents=self.mask_latents, num_inference_steps=num_inference_steps)


        if not output_type == "latent":
            image = self.vae.decode(x0 / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()
        return image        
            

    def save_latent_2_image(self, latent, save_path, output_type='pil'):
        latent = latent.to(self.device)
        image = self.vae.decode(latent / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)[0]
        image.save(save_path)


    def prepare_image_latents(self, image, batch_size, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:
            latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0)
            else:
                latents = self.vae.encode(image).latent_dist.sample(generator)

            latents = self.vae.config.scaling_factor * latents

        if batch_size != latents.shape[0]:
            if batch_size % latents.shape[0] == 0:
                # expand image_latents for batch_size
                deprecation_message = (
                    f"You have passed {batch_size} text prompts (`prompt`), but only {latents.shape[0]} initial"
                    " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                    " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                    " your script to pass as many initial images as text prompts to suppress this warning."
                )
                deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
                additional_latents_per_image = batch_size // latents.shape[0]
                latents = torch.cat([latents] * additional_latents_per_image, dim=0)
            else:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {latents.shape[0]} to {batch_size} text prompts."
                )
        else:
            latents = torch.cat([latents], dim=0)

        return latents
    
    @torch.no_grad()
    def invert(
            self,
            prompt: Optional[str] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "latent",
            prompt_embeds: Optional[torch.FloatTensor] = None,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            start_step:int = 0,
            end_step:int = 50,
            save_latens : bool =False,
    ):

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        latents = latents
        

        replace_to_back = []
        if save_latens:
            replace_to_back.append(latents.to('cpu'))

        num_images_per_prompt = 1
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
        )

        # Prepare timesteps
        timesteps = self.inverse_scheduler.timesteps
        # Denoising loop where we obtain the cross-attention maps.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.inverse_scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                index = i
                if index < start_step:
                    continue
                if index >= end_step:
                    break
       
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.inverse_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.inverse_scheduler.step(noise_pred, t, latents).prev_sample

                if save_latens:
                    replace_to_back.append(latents.to('cpu'))

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.inverse_scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        inverted_latents = latents.detach().clone()


        if save_latens:
            print(f"save intermediate inversion from_{start_step}_to_{end_step},total:{len(replace_to_back)}")
            return inverted_latents,replace_to_back
        else:
            return inverted_latents
        
    @torch.no_grad()
    def inject_noise(self,
                     x_t: torch.Tensor,
                     end_step: int,
                     start_step: int,
                     ):  

        a_t = self.scheduler.alphas_cumprod[self.inverse_scheduler.timesteps[start_step]]
        a_tm = self.scheduler.alphas_cumprod[self.inverse_scheduler.timesteps[end_step]]
        noise = torch.randn_like(x_t)
        x_tm = ((a_tm / a_t) ** 0.5) * x_t + ((1 - (a_tm / a_t)) ** 0.5) * noise
        return x_tm

    def sample_and_replace_with_mask(self, x, last_step, end_step, back_step, replace_step_list, eta, mask_latents, prompt,
                           generator, Vs_back_to_front, bg_mask, num_inference_steps):
        for k in replace_step_list:  # 7, 6
            x = self.sample_with_mask(prompt=prompt, eta=eta, generator=generator, latents=x, 
                                      start_step=last_step, end_step=k, bg_mask=bg_mask, num_inference_steps=num_inference_steps)
            last_step = k
            replace_index = back_step - k  

            for i, V_back_to_front in enumerate(Vs_back_to_front):
                x = self.replace_with_seed(bg_noise=x, gt_noise=V_back_to_front[replace_index], mask_latent=mask_latents[i], xy=self.xys[i])
        x = self.sample_with_mask(prompt=prompt, eta=eta, generator=generator, latents=x, 
                                  start_step=last_step, end_step=end_step,bg_mask=bg_mask, num_inference_steps=num_inference_steps)
        return x
    
    def replace_with_seed(self, bg_noise, gt_noise, mask_latent, xy):
        x_offset, y_offset = xy

        x_offset = x_offset // self.vae_scale_factor
        y_offset = y_offset // self.vae_scale_factor

        seed_size_latent = gt_noise.shape[-1]
        gt_noise = gt_noise.to(self.device)
        bg_noise[:,:,y_offset:y_offset+seed_size_latent,x_offset:x_offset+seed_size_latent][mask_latent==1] = gt_noise[mask_latent==1]
        return bg_noise
    
    @torch.no_grad()
    def sample_with_mask(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "latent",
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            start_step: int = 50,
            end_step: int = 0,
            bg_mask : torch.Tensor = None,
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt

        neg_prompt = "a bad quality and low resolution image,extra fingers,deformed hands"
        diverse_prompt = "a grassy background"
        suppress_prompt = "foreground objects"

        prompt_neg_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=neg_prompt,
        )
        div_sup_embeds = self._encode_prompt(
            diverse_prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=suppress_prompt,
        )


        # 4. Prepare timesteps

        timesteps = self.scheduler.timesteps


        num_channels_latents = self.unet.config.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_neg_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
   
        for i, t in enumerate(timesteps):

            index = len(timesteps) - i - 1  # 49
            if index >= start_step:
                continue
            if index < end_step:
                break
            latent_model_input = torch.cat([latents] * 4) if do_classifier_free_guidance else latents

            prompt_embeds = torch.cat([prompt_neg_embeds, div_sup_embeds])

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_neg, noise_pred_text, noise_pred_sup, noise_pred_div = noise_pred.chunk(4)

                bg_mask = bg_mask.to(device)

                all_one_mask = torch.ones_like(bg_mask).to(device)
                background_mask = all_one_mask - bg_mask

                noise_pred_total_neg = bg_mask * noise_pred_neg + background_mask * (noise_pred_neg + noise_pred_sup - noise_pred_div)
                noise_pred = noise_pred_total_neg + guidance_scale * (noise_pred_text - noise_pred_total_neg)


            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)



        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)

        return image


    def sample_and_replace(self, x, last_step, end_step, back_step, replace_step_list, eta, mask_latents, prompt,
                           generator, Vs_back_to_front, num_inference_steps):
        for k in replace_step_list:  
            x = self.sample(prompt=prompt, eta=eta, generator=generator, latents=x, start_step=last_step, end_step=k, num_inference_steps=num_inference_steps)
            last_step = k
            replace_index = back_step - k  
            for i, V_back_to_front in enumerate(Vs_back_to_front):
                x = self.replace_with_seed(bg_noise=x, gt_noise=V_back_to_front[replace_index], mask_latent=mask_latents[i], xy=self.xys[i])
        x = self.sample(prompt=prompt, eta=eta, generator=generator, latents=x, start_step=last_step, end_step=end_step, num_inference_steps=num_inference_steps)
        return x
    
    @torch.no_grad()
    def sample(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "latent",
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            start_step: int = 50,
            end_step: int = 0,
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                index = len(timesteps) - i - 1  
                if index >= start_step:
                    continue
                if index < end_step:  
                    break

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)

        return image