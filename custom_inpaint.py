import torch
import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from typing import Union, List, Optional, Callable

class CustomInpaintPipeline(StableDiffusionInpaintPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: torch.FloatTensor,
        mask_image: torch.FloatTensor,
        strength: float = 1.0,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        last_grad_steps: int = 0,
        **kwargs,
    ):
        # 1. Check inputs and define parameters
        height, width = image.shape[-2:]
        self.check_inputs(prompt, image, mask_image, height, width, strength, callback_steps, negative_prompt, None, None)
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # 2. Encode prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 3. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        
        # 4. Prepare latents for INPAINTING
        image_for_vae = image * 2.0 - 1.0
        original_image_latents = self.vae.encode(image_for_vae).latent_dist.sample(generator=generator)
        original_image_latents = self.vae.config.scaling_factor * original_image_latents

        # Denoising을 시작할 초기 노이즈 생성
        noise = torch.randn(original_image_latents.shape, generator=generator, device=device, dtype=text_embeddings.dtype)
        
        # ❗ FIX: timesteps 배열 전체가 아닌, 가장 첫 번째 timestep[0]만 사용합니다.
        latents = self.scheduler.add_noise(original_image_latents, noise, timesteps[0])
        
        # 5. Prepare mask
        mask = F.interpolate(mask_image, size=(height // self.vae_scale_factor, width // self.vae_scale_factor))
        mask = (mask > 0.5).to(dtype=text_embeddings.dtype)

        num_timesteps = len(timesteps)
        grad_start_step = num_timesteps - last_grad_steps

        # 6. Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            if do_classifier_free_guidance:
                current_mask = torch.cat([mask] * 2)
                current_original_image_latents = torch.cat([original_image_latents] * 2)
            else:
                current_mask = mask
                current_original_image_latents = original_image_latents
            
            latent_model_input = torch.cat([latent_model_input, current_mask, current_original_image_latents], dim=1)

            if i < grad_start_step or last_grad_steps == 0:
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings, return_dict=False)[0]
            else:
                with torch.enable_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings, return_dict=False)[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            step_output = self.scheduler.step(noise_pred, t, latents, generator=generator, return_dict=False)
            
            if i < grad_start_step or last_grad_steps == 0:
                latents = step_output[0]
            else:
                with torch.enable_grad():
                    latents = step_output[0].clone()

        # 7. Post-processing
        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == 'tensor':
            latents = 1 / self.vae.config.scaling_factor * latents
            image = self.vae.decode(latents, return_dict=False)[0]
            image = ((image + 1) / 2).clamp(0, 1)
            has_nsfw_concept = None
        else:
            latents = 1 / self.vae.config.scaling_factor * latents
            image = self.vae.decode(latents, return_dict=False)[0]

            # safety checker는 PIL 이미지가 필요하므로 여기서 실행
            image_for_safety_checker = self.image_processor.postprocess(image, output_type="pil")
            safety_checker_output = self.run_safety_checker(image_for_safety_checker, device, text_embeddings.dtype)
            has_nsfw_concept = safety_checker_output.nsfw_content_detected
            
            # 최종 이미지 정규화
            image = ((image + 1) / 2).clamp(0, 1)

        # 8. Convert to PIL
        # output_type이 "pil"이고, "latent"가 아닐 때만 변환
        if output_type == "pil" and image.shape[1] == 3:
             image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
