"""Video export utilities for painting animations.

This module provides functions to export the painting process as video
animations, showing brush strokes being iteratively drawn on canvas with
a polished visual presentation including shadows and textured backgrounds.

Memory-efficient implementation: frames are streamed directly to the video
encoder instead of being stored in memory, allowing export of animations
with many thousands of brush strokes.
"""
from collections.abc import Generator, Iterable, Sequence
from pathlib import Path

import av
import torch
import torch.nn.functional as F
from torch import Tensor

from painting.brushes import Brush


def _create_canvas_texture(
    size: tuple[int, int],
    color: tuple[int, int, int] = (240, 235, 230),
    noise_strength: float = 3.0,
    device: torch.device | None = None,
) -> Tensor:
    width, height = size
    texture = torch.zeros(3, height, width, device=device)
    for c, val in enumerate(color):
        texture[c] = val

    # Add subtle noise for texture effect
    noise = torch.randn(1, height, width, device=device) * noise_strength
    texture = texture + noise
    return texture.clamp(0, 255)


def _create_gaussian_kernel(blur_radius: int, device: torch.device) -> Tensor:
    kernel_size = blur_radius * 2 + 1
    sigma = blur_radius / 3.0
    x_coord = torch.arange(kernel_size, device=device).float() - blur_radius
    gaussian_1d = torch.exp(-x_coord ** 2 / (2 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
    return gaussian_2d.unsqueeze(0).unsqueeze(0)


def _draw_shadow(
    background: Tensor,
    x: int,
    y: int,
    width: int,
    height: int,
    offset: int = 8,
    blur_radius: int = 15,
    shadow_color: tuple[int, int, int] = (50, 50, 50),
    shadow_alpha: float = 0.3,
    gaussian_kernel: Tensor | None = None,
    shadow_rgb: Tensor | None = None,
) -> Tensor:
    """Draws a blurred shadow on the background at the specified position."""
    bg_h, bg_w = background.shape[1], background.shape[2]

    shadow_mask = torch.zeros(1, bg_h, bg_w, device=background.device)

    # Shadow rectangle (offset from object)
    sx = x + offset
    sy = y + offset
    sx_end = min(sx + width, bg_w)
    sy_end = min(sy + height, bg_h)
    sx = max(0, sx)
    sy = max(0, sy)

    if sx < sx_end and sy < sy_end:
        shadow_mask[0, sy:sy_end, sx:sx_end] = 1.0

    # Apply Gaussian blur to shadow
    if blur_radius > 0:
        if gaussian_kernel is None:
            gaussian_kernel = _create_gaussian_kernel(blur_radius, background.device)

        shadow_mask = shadow_mask.unsqueeze(0)
        shadow_mask = F.pad(shadow_mask, [blur_radius] * 4, mode="reflect")
        shadow_mask = F.conv2d(shadow_mask, gaussian_kernel)
        shadow_mask = shadow_mask.squeeze(0)

    if shadow_rgb is None:
        shadow_rgb = torch.tensor(shadow_color, device=background.device).float().view(3, 1, 1)
    shadow_mask = shadow_mask * shadow_alpha
    background = background * (1 - shadow_mask) + shadow_rgb * shadow_mask

    return background


def _paste_image(
    background: Tensor,
    image: Tensor,
    x: int,
    y: int,
) -> Tensor:
    """Pastes an image onto background at specified position."""
    bg_h, bg_w = background.shape[1], background.shape[2]
    img_h, img_w = image.shape[1], image.shape[2]

    x_end = min(x + img_w, bg_w)
    y_end = min(y + img_h, bg_h)
    x_start = max(0, x)
    y_start = max(0, y)

    src_x_start = x_start - x
    src_y_start = y_start - y
    src_x_end = src_x_start + (x_end - x_start)
    src_y_end = src_y_start + (y_end - y_start)

    if x_start < x_end and y_start < y_end:
        background[:, y_start:y_end, x_start:x_end] = image[:, src_y_start:src_y_end, src_x_start:src_x_end]

    return background


def _resize_image(image: Tensor, size: tuple[int, int]) -> Tensor:
    needs_batch = len(image.shape) == 3
    if needs_batch:
        image = image.unsqueeze(0)

    resized = F.interpolate(image, size=size, mode="bilinear", align_corners=False)

    if needs_batch:
        resized = resized.squeeze(0)

    return resized


def _compose_frame(
    canvas: Tensor,
    target: Tensor | Sequence[Tensor] | None = None,
    layout: str = "none",
    frame_size: tuple[int, int] = (1280, 720),
    margin: int = 40,
    gap: int = 30,
    shadow_offset: int = 8,
    shadow_blur: int = 15,
    background_color: tuple[int, int, int] = (240, 235, 230),
    corner_scale: float = 0.25,
    cached_background: Tensor | None = None,
    gaussian_kernel: Tensor | None = None,
    gaussian_kernel_small: Tensor | None = None,
    shadow_rgb: Tensor | None = None,
    output_buffer: Tensor | None = None,
) -> Tensor:
    """Composes canvas and optional target images onto a textured background.

    Layout options: "none" (canvas only), "side_by_side", or "corner".
    """
    frame_w, frame_h = frame_size
    canvas_h, canvas_w = canvas.shape[1], canvas.shape[2]
    device = canvas.device

    if target is not None and not isinstance(target, (list, tuple)):
        targets = [target]
    else:
        targets = list(target) if target else []

    if cached_background is not None:
        if output_buffer is not None:
            output_buffer.copy_(cached_background)
            background = output_buffer
        else:
            background = cached_background.clone()
    else:
        background = _create_canvas_texture(frame_size, background_color, device=device)

    if layout == "none" or not targets:
        available_w = frame_w - 2 * margin
        available_h = frame_h - 2 * margin

        # Scale canvas to fit while maintaining aspect ratio
        scale = min(available_w / canvas_w, available_h / canvas_h)
        display_w = int(canvas_w * scale)
        display_h = int(canvas_h * scale)

        display_canvas = _resize_image(canvas, (display_h, display_w))

        x = (frame_w - display_w) // 2
        y = (frame_h - display_h) // 2

        background = _draw_shadow(background, x, y, display_w, display_h, shadow_offset, shadow_blur,
                                  gaussian_kernel=gaussian_kernel, shadow_rgb=shadow_rgb)
        background = _paste_image(background, display_canvas, x, y)

    elif layout == "side_by_side":
        # canvas + targets
        num_images = 1 + len(targets)
        total_gaps = gap * (num_images - 1)
        available_w = frame_w - 2 * margin - total_gaps
        available_h = frame_h - 2 * margin

        # Each image gets equal share of width
        max_image_w = available_w // num_images
        scale = min(max_image_w / canvas_w, available_h / canvas_h)
        display_size = int(min(canvas_w, canvas_h) * scale)

        display_canvas = _resize_image(canvas, (display_size, display_size))

        display_targets = [
            _resize_image(t.to(device), (display_size, display_size))
            for t in targets
        ]

        total_width = display_size * num_images + total_gaps
        x = (frame_w - total_width) // 2
        y = (frame_h - display_size) // 2

        background = _draw_shadow(background, x, y, display_size, display_size, shadow_offset, shadow_blur,
                                  gaussian_kernel=gaussian_kernel, shadow_rgb=shadow_rgb)
        background = _paste_image(background, display_canvas, x, y)
        x += display_size + gap

        for display_target in display_targets:
            background = _draw_shadow(background, x, y, display_size, display_size, shadow_offset, shadow_blur,
                                      gaussian_kernel=gaussian_kernel, shadow_rgb=shadow_rgb)
            background = _paste_image(background, display_target, x, y)
            x += display_size + gap

    elif layout == "corner":
        available_w = frame_w - 2 * margin
        available_h = frame_h - 2 * margin

        num_targets = len(targets)
        # Total gaps for targets: gap between canvas and first target, plus gaps between targets
        total_target_gaps = gap * num_targets if num_targets > 0 else 0

        # Determine if we need to scale down to fit
        # Account for targets on right: available_w needs room for canvas + gaps + all targets
        if num_targets > 0:
            max_canvas_w_for_width = (available_w - total_target_gaps) / (1 + num_targets * corner_scale)
        else:
            max_canvas_w_for_width = available_w
        max_canvas_h_for_height = available_h

        # Scale factor: only scale down if needed, never scale up
        scale = min(1.0, max_canvas_w_for_width / canvas_w, max_canvas_h_for_height / canvas_h)

        display_w = int(canvas_w * scale)
        display_h = int(canvas_h * scale)

        # Resize canvas only if scale < 1
        if scale < 1.0:
            display_canvas = _resize_image(canvas, (display_h, display_w))
        else:
            display_canvas = canvas

        target_size = int(min(display_h, display_w) * corner_scale)

        total_width = display_w + num_targets * (gap + target_size)

        start_x = (frame_w - total_width) // 2

        canvas_x = start_x

        max_height = max(display_h, target_size) if num_targets > 0 else display_h
        bottom_y = (frame_h + max_height) // 2

        canvas_y = bottom_y - display_h

        background = _draw_shadow(background, canvas_x, canvas_y, display_w, display_h,
                                  shadow_offset, shadow_blur,
                                  gaussian_kernel=gaussian_kernel, shadow_rgb=shadow_rgb)
        background = _paste_image(background, display_canvas, canvas_x, canvas_y)

        target_x = start_x + display_w + gap
        target_y = bottom_y - target_size
        for t in targets:
            display_target = _resize_image(t.to(device), (target_size, target_size))
            background = _draw_shadow(background, target_x, target_y, target_size, target_size,
                                      shadow_offset // 2, shadow_blur // 2,
                                      gaussian_kernel=gaussian_kernel_small, shadow_rgb=shadow_rgb)
            background = _paste_image(background, display_target, target_x, target_y)
            target_x += target_size + gap

    return background.clamp(0, 255)


def _generate_painting_frames(
    brush: Brush,
    brush_params: Tensor,
    initial_canvas: Tensor,
    brush_stroke_rendering_batch_size: int = 5,
) -> Generator[tuple[Tensor, int], None, None]:
    """Yields (canvas, stroke_index) tuples, one per brush stroke drawn."""
    device = brush_params.device
    canvas = initial_canvas.clone().to(device)

    yield canvas, -1

    for stroke_idx in range(len(brush_params)):
        params = brush_params[stroke_idx]
        canvas = brush.draw_on_single_canvas(
            brush_params=params,
            canvas=canvas,
            rendering_batch_size=brush_stroke_rendering_batch_size,
        )
        yield canvas, stroke_idx


def _compute_frame_duration_ms(
    stroke_idx: int,
    speed_schedule: Sequence[float] | None = None,
    speed_milestones: Sequence[int] | None = None,
    default_strokes_per_second: float = 10.0,
) -> float:
    """Computes frame duration in milliseconds based on speed schedule and milestones."""
    if speed_schedule is not None:
        if speed_milestones is not None:
            # Find which segment this stroke belongs to
            segment_idx = 0
            for milestone in speed_milestones:
                if stroke_idx >= milestone:
                    segment_idx += 1
                else:
                    break
            # Use the speed for this segment (clamped to schedule length)
            segment_idx = min(segment_idx, len(speed_schedule) - 1)
            strokes_per_second = speed_schedule[segment_idx]
        else:
            # No milestones - use first speed for all strokes
            strokes_per_second = speed_schedule[0]
    else:
        strokes_per_second = default_strokes_per_second

    # Clamp to reasonable range
    strokes_per_second = max(0.5, min(strokes_per_second, 1000.0))

    return 1000.0 / strokes_per_second


def _write_av_frame(stream: av.VideoStream, container: av.container.OutputContainer, frame_np) -> None:
    av_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
    for packet in stream.encode(av_frame):
        container.mux(packet)


def _save_video(
    frames: Iterable[tuple[Tensor, int]],
    output_path: str,
    fps: int = 30,
    speed_schedule: Sequence[float] | None = None,
    speed_milestones: Sequence[int] | None = None,
    default_strokes_per_second: float = 10.0,
    hold_final_seconds: float = 1.0,
    target: Tensor | Sequence[Tensor] | None = None,
    target_layout: str = "none",
    frame_size: tuple[int, int] = (1280, 720),
    codec: str = "libx264",
    crf: int = 17,
    preset: str = "slow",
    shadow_blur: int = 15,
    background_color: tuple[int, int, int] = (240, 235, 230),
    shadow_color: tuple[int, int, int] = (50, 50, 50),
) -> None:
    """Streams painting frames to an MP4 video file with variable speed support."""
    frame_time_ms = 1000.0 / fps

    # Pre-compute cached resources (will be initialized on first frame)
    cached_background: Tensor | None = None
    gaussian_kernel: Tensor | None = None
    gaussian_kernel_small: Tensor | None = None
    shadow_rgb: Tensor | None = None
    compose_buffer: Tensor | None = None
    frame_cpu_buffer: Tensor | None = None

    # Pre-move targets to device once (will be done on first frame when device is known)
    targets_on_device: list[Tensor] | None = None

    frame_w, frame_h = frame_size

    container = av.open(output_path, mode="w")
    stream = container.add_stream(codec, rate=fps)
    stream.width = frame_w
    stream.height = frame_h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf), "preset": preset}

    # Accumulate time to handle fast speeds where multiple strokes fit in one frame
    accumulated_time_ms = 0.0
    last_frame_np = None
    last_canvas = None

    for canvas, stroke_idx in frames:
        if stroke_idx < 0:
            stroke_idx = 0

        device = canvas.device

        # Initialize cached resources on first frame (when device is known)
        if cached_background is None:
            cached_background = _create_canvas_texture(frame_size, background_color, device=device)
            gaussian_kernel = _create_gaussian_kernel(shadow_blur, device)
            gaussian_kernel_small = _create_gaussian_kernel(shadow_blur // 2, device)
            shadow_rgb = torch.tensor(shadow_color, device=device).float().view(3, 1, 1)
            compose_buffer = torch.empty(3, frame_h, frame_w, device=device)
            frame_cpu_buffer = torch.empty(frame_h, frame_w, 3, dtype=torch.uint8, device="cpu")

            # Pre-move targets to device
            if target is not None:
                if isinstance(target, (list, tuple)):
                    targets_on_device = [t.to(device) for t in target]
                else:
                    targets_on_device = [target.to(device)]

        # Accumulate time for this stroke
        duration_ms = _compute_frame_duration_ms(
            stroke_idx, speed_schedule, speed_milestones, default_strokes_per_second
        )
        accumulated_time_ms += duration_ms
        last_canvas = canvas

        # Only compose and write when we've accumulated at least one frame's worth of time
        if accumulated_time_ms >= frame_time_ms:
            # Clone canvas since generator may mutate it in-place
            composed = _compose_frame(
                canvas.clone(),
                target=targets_on_device,
                layout=target_layout,
                frame_size=frame_size,
                cached_background=cached_background,
                gaussian_kernel=gaussian_kernel,
                gaussian_kernel_small=gaussian_kernel_small,
                shadow_rgb=shadow_rgb,
                output_buffer=compose_buffer,
            )

            # Convert to numpy uint8 [H, W, 3] using pre-allocated CPU buffer
            frame_cpu_buffer.copy_(composed.permute(1, 2, 0).clamp(0, 255).byte())
            frame_np = frame_cpu_buffer.numpy()
            last_frame_np = frame_np

            # Write as many frames as accumulated time allows
            while accumulated_time_ms >= frame_time_ms:
                _write_av_frame(stream, container, frame_np)
                accumulated_time_ms -= frame_time_ms

    # Handle case where final strokes didn't accumulate enough time to write
    if last_canvas is not None and (last_frame_np is None or accumulated_time_ms > 0):
        composed = _compose_frame(
            last_canvas,
            target=targets_on_device,
            layout=target_layout,
            frame_size=frame_size,
            cached_background=cached_background,
            gaussian_kernel=gaussian_kernel,
            gaussian_kernel_small=gaussian_kernel_small,
            shadow_rgb=shadow_rgb,
            output_buffer=compose_buffer,
        )
        frame_cpu_buffer.copy_(composed.permute(1, 2, 0).clamp(0, 255).byte())
        frame_np = frame_cpu_buffer.numpy()
        _write_av_frame(stream, container, frame_np)
        last_frame_np = frame_np

    if last_frame_np is not None:
        hold_frames = int(hold_final_seconds * fps)
        for _ in range(hold_frames):
            _write_av_frame(stream, container, last_frame_np)

    # Flush encoder and close
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def create_painting_video(
    brush: Brush,
    brush_params: Tensor,
    initial_canvas: Tensor,
    output_path: str | Path,
    target_images: Tensor | Sequence[Tensor] | None = None,
    target_layout: str = "corner",
    brush_stroke_rendering_batch_size: int = 5,
    frame_size: tuple[int, int] = (1280, 720),
    fps: int = 60,
    speed_schedule: Sequence[float] | None = None,
    speed_milestones: Sequence[int] | None = None,
    default_strokes_per_second: float = 20.0,
    hold_final_seconds: float = 5.0,
    codec: str = "libx264",
    crf: int = 17,
    preset: str = "slow",
) -> None:
    """Creates a painting animation video.

    This is a convenience function that combines frame generation and export.

    Memory-efficient: frames are streamed directly to the output file instead
    of being stored in memory, allowing export of animations with many
    thousands of brush strokes.

    Note: The canvas size is determined by brush.canvas_size. The final frame
    size can be controlled via the frame_size parameter - the canvas will be
    scaled to fit within the frame while maintaining aspect ratio.

    Speed scheduling works similarly to PyTorch's MultiStepLR: milestones define
    the stroke indices where the speed changes, and schedule defines the speeds
    for each segment.

    Args:
        brush: Brush instance to use for rendering.
        brush_params: Tensor of shape [N, P] with normalized brush parameters.
        initial_canvas: Starting canvas tensor of shape [3, H, W] to paint on.
            Use a blank canvas (e.g., torch.ones(3, H, W) * 255 for white) to
            start fresh, or provide a pre-existing canvas to continue painting.
        output_path: Path to save the video.
        target_images: Optional target image(s) for comparison display.
            Can be a single tensor [3, H, W] or a sequence of tensors.
        target_layout: Layout for targets - "none", "side_by_side", or "corner".
        frame_size: Output frame size (width, height).
        fps: Frames per second.
        speed_schedule: List of strokes per second for each segment.
            If provided with milestones, defines speeds for segments:
            [0, milestone[0]), [milestone[0], milestone[1]), etc.
            If provided without milestones, uses first value for all strokes.
        speed_milestones: List of stroke indices where speed changes.
        default_strokes_per_second: Default speed when no schedule is provided.
        hold_final_seconds: How long to hold the final frame.
        codec: Video codec (default "libx264" for high quality and compatibility).
        crf: Constant Rate Factor for quality (0-51, lower = better, default 17
            for near visually lossless).
        preset: Encoding preset ("slow" for better compression, "fast" for speed).
    """
    # Generate frames as a streaming generator.
    frames = _generate_painting_frames(
        brush=brush,
        brush_params=brush_params,
        initial_canvas=initial_canvas,
        brush_stroke_rendering_batch_size=brush_stroke_rendering_batch_size,
    )

    # Save as MP4 (streaming, memory efficient)
    _save_video(
        frames=frames,
        output_path=str(output_path),
        fps=fps,
        speed_schedule=speed_schedule,
        speed_milestones=speed_milestones,
        default_strokes_per_second=default_strokes_per_second,
        hold_final_seconds=hold_final_seconds,
        target=target_images,
        target_layout=target_layout,
        frame_size=frame_size,
        codec=codec,
        crf=crf,
        preset=preset,
    )
