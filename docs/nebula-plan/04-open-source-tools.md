# Open-source tools

Everything below is free. The ones marked "core" save real work. Check each licence before copying code: MIT and BSD code can go into a GPL project; non-commercial (NC) licences shouldn't be copied at all.

| Tool | What it does | Use it for | Licence | Priority |
| --- | --- | --- | --- | --- |
| [Phantom `set_velfield_from_cubes`](https://phantomsph.readthedocs.io/en/latest/api/setup/velfield_fromcubes.html) | Sets particle velocities from turbulent velocity cubes, with amplitude, radius and an edge fall-off | Turbulent starts for Act 1 and Act 2, inside Phantom itself | GPL-3 (Phantom) | Core |
| [Phantom turbulence cube files](https://zenodo.org/records/13162515) | Ready-made turbulent velocity cubes that Phantom downloads automatically | A first test without generating your own field | Open data | Core |
| [TurbGen](https://www.mso.anu.edu.au/~chfeder/codes/turbgen/turbgen.html) (Federrath) | Generates turbulent velocity fields, including single realisations, with chosen spectrum and solenoidal/compressive mix | Making your own k⁻⁴ fields | Check the repo | Useful |
| [pyFC](https://pypi.org/project/pyFC) | Lognormal "fractal cubes" of density (Lewis & Austin 2002); periodic, so they tile | Lumpy density starts; density detail for the backdrop | Check the repo | Core |
| [PyNeb](https://research.iac.es/proyecto/PyNeb/pyneb.html) | Emission-line emissivities vs temperature and density, for recombination and collisional lines | Pre-compute small T × n tables for Hα, Hβ, [O III], [N II], [S II]; the shader just reads them | Check the repo | Core |
| [dust_extinction](https://pypi.org/project/dust-extinction/1.3) (Gordon) | Published dust extinction curves in one place | Extinction at each line's wavelength, baked in as shader constants | BSD-3 | Core |
| [psrdnoise](https://awesome.ecosyste.ms/projects/github.com%2Fstegu%2Fpsrdnoise) (Gustavson & McEwan) | Tiling 3D simplex noise and flow noise with analytic gradients; includes a WGSL port | Rest-field wisps; curl noise from the gradients | Check headers | Core |
| [wgsl-noise](https://npmjs.com/package/wgsl-noise) | Classic Perlin, simplex and cellular noise ported to WGSL | Worley/cellular noise for bubbles | MIT | Useful |
| [SPLASH](https://github.com/danieljprice/splash) (Price) | The standard SPH visualizer; renders with the SPH kernel and reads Phantom dumps | Cross-checking your grid deposition and quick looks at Act 1 runs | Open source | Useful |
| [yt volume rendering](https://yt-project.org/doc/visualizing/volume_rendering.html) | CPU volume rendering with emission/absorption transfer functions | An independent check of your colour transfer on a test grid | Open source | Optional |
| [OpenSpace](https://www.Github.com/OpenSpace/OpenSpace) | AMNH/Linköping planetarium software with interactive volume rendering | Seeing how planetariums stage the Galactic Centre | Open source, non-commercial | Inspiration |
| [Gaia Sky nebulae](https://tonisagrista.com/blog/2024/rendering-aurorae-nebulae/) | Real-time ray-marched nebula shaders | Look-dev reference | Based on CC BY-NC-SA shaders: don't copy | Inspiration |
| [Mesa lavapipe](https://motrixsim.readthedocs.io/en/latest/user_guide/render/headless_cpu.html) | Vulkan on the CPU, which wgpu picks up automatically | Running the renderer without a GPU | Open source (Mesa) | Core for free rendering |

## How they fit together

1. **Starts:** pyFC density cube + TurbGen or Phantom cube velocities → Phantom setup.
2. **Simulation:** Phantom (Act 1 and Act 2) + your orbit integrator (gravity-only particles).
3. **Colour tables, built offline:** PyNeb line emissivities + dust_extinction curves + your CIE table → small lookup textures.
4. **Rendering:** your WGSL marcher + psrdnoise for wisps; lavapipe wherever there's no GPU.
5. **Checks:** SPLASH and yt as independent renderers for tests.
