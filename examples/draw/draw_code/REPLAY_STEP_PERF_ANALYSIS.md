# Replay Step Performance Analysis

Date: 2026-04-25

Scope:
- Script: `examples/draw/draw_code/replay.py`
- Goal: explain the current replay step bottlenecks after the cleanup work

## Current Status

The replay loop has already been simplified:

- per-step file saving was removed
- viewport capture was removed
- UV projection was removed
- the loop now runs as a local `while not render.is_closed`
- scene overrides now use only in-memory `motrixsim.msd` mutation

So the current slowdown is no longer caused by disk IO or capture flow.

## Measured Breakdown

Recent task 04 smoke runs showed roughly:

- `env_step`: `33-48 ms`
- `gs_render`: `55-69 ms`
- `tex_upload`: `~0.01-0.02 ms`
- `render_sync`: `~0.02-0.03 ms`

Practical conclusion:

1. the two dominant costs are `env.step(...)` and GS rendering
2. texture upload and `render.sync(...)` are negligible right now

## Main Causes

### 1. `env.step(...)` is still expensive

The replay loop prints `env_step` timing directly around:

- `env.step(a_batch)`

Earlier instrumentation inside the env path showed that `update_state(...)` is a major part of this cost.

### 2. `update_state(...)` likely includes pixel rendering/readback

The current env pipeline appears to build observations during the final substep.
That path likely reaches image rendering and CPU-side pixel materialization inside env observation construction.

This is the main reason the simulation side is still tens of milliseconds per step even after capture removal.

### 3. GS rendering is the other dominant cost

Each step still does:

- `forward_kinematic(model, data)`
- `model.get_link_poses(data)`
- `gs_renderer.batch_update_gaussians(...)`
- `gs_renderer.batch_env_render(...)`

That is now the heaviest graphics workload in the loop.

### 4. GS output still performs GPU to CPU readback

The frustum-screen texture path still converts the rendered tensor to `numpy`:

```python
rgb_t.detach().cpu().numpy()
```

So each step still pays for a synchronization point and full image readback before assigning:

```python
gs_screen_img.pixels = gs_u8[0]
```

With the current Motrix texture API, this readback does not look easy to avoid.

## Current Practical Conclusion

The bottlenecks are now:

1. `env.step(...)`, especially `update_state(...)`
2. `render_gs_batch_rgb_u8(...)`

The old XML-temp path, viewport capture path, and UV path are no longer relevant to current step speed.

## Next Investigation Targets

If replay needs to become faster, the next useful checks are:

1. inspect `env.step -> update_state -> _build_obs -> _render_pixels`
2. determine whether replay can skip image observations entirely
3. evaluate whether GS preview can be disabled or downscaled for fast replay mode
