# Free compute

Four free sources are enough for this project: your Mac, Kaggle, the rest of your Google Cloud trial credit, and Mesa's lavapipe so the renderer runs on CPU-only machines. Options that need a faculty sponsor are out for now, and a few popular "free" routes break their terms for this kind of use.

## Use these

| Option | What you get | Use it for | Catch |
| --- | --- | --- | --- |
| Your Mac | 2-core Intel, Iris Plus GPU | Look-dev, 50k test runs, final 4K overnights | Its GPU has crashed on long renders before. Keep adaptive tiles on, lid open, plugged in. |
| Google Cloud trial (what's left) | Your remaining credit; 8-vCPU cap; ends ~Jan 16, 2027 | Act 1 production run in December, then the Act 2 cloud run | It's your one trial: the free trial is [only for new customers](https://cloud.google.com/terms/trial). Never click "Upgrade" by accident. |
| [Kaggle GPU notebooks](https://gpuperhour.com/blog/free-cloud-gpus-and-credits) | ~30 GPU-hours/week (P100 or 2× T4); 12-hour sessions; 20 GB saved per notebook | Render chunks of the final, short clips | The quota floats. Headless Vulkan is untested there. 20 GB won't hold the 4K frames, so encode per session. |
| Kaggle CPU notebooks | [4 cores, 12-hour sessions, 20 GB saved](https://packtpub.com/en-mx/product/the-kaggle-book-9781801817479/chapter/working-and-learning-with-kaggle-notebooks-4/section/working-and-learning-with-kaggle-notebooks-4) | Chaining Phantom runs from restart dumps; CPU rendering with lavapipe | About half the speed of the 8-thread VM (my estimate), so roughly twice as many hours. |
| [Mesa lavapipe](https://motrixsim.readthedocs.io/en/latest/user_guide/render/headless_cpu.html) | Vulkan on the CPU. Once installed, wgpu picks it as a CPU adapter, with the same result as the GPU, only slower. | Running your renderer anywhere: Kaggle CPU, a spare laptop, a university machine | Much slower than a GPU; measure frames per hour before relying on it. |
| [Azure for Students](https://azure.microsoft.com/pricing/offers/ms-azr-0170p/) | $100 credit for 12 months, no card; full-time student, verified by school email | A backup run if the Google credit runs out | One per person, while supplies last. |

## Worth one application or email

- **[Open Science Pool (OSPool)](https://osg-htc.org/services/ospool-registration.html):** free computing for [any researcher affiliated with a project at a US-based academic institution](https://osg-htc.org/services/ospool), built for many independent jobs. Your rendering is a perfect fit: every frame is its own job, run with lavapipe on CPUs. Sign-up is an application plus a short consultation call. Undergrad eligibility without a faculty project is unclear, so describe the DRP project honestly and ask. Long single runs like Phantom are a poor fit there.
- **UC Davis HPC:** Hive's free tier is [listed for staff, faculty and graduate students](https://docs.hpc.ucdavis.edu/hive/). The Farm cluster has a [free tier for users without a PI](https://docs.hpc.ucdavis.edu/general/account-requests/). Email hpc-help@ucdavis.edu to ask whether undergrads can use it.

## Not possible right now, or avoid

| Option | Why not |
| --- | --- |
| NSF ACCESS | [Undergrads can't lead a project](https://allocations.access-ci.org/allocations-policy); a grad student can lead one only for their own thesis work. Needs a faculty PI. |
| GitHub Actions as a render farm | GitHub's terms bar using hosted runners for [activity unrelated to the production, testing, deployment, or publication of the software project](https://docs.github.com/en/site-policy/github-terms/github-terms-for-additional-products-and-features). Codespaces has the same kind of rule. |
| DigitalOcean student credit | The GitHub Student Pack offer [ended July 31, 2026](https://www.getaiperks.com/en/ai/digitalocean-students). |
| Oracle Always Free | [Cut to 2 Arm cores and 12 GB](https://infoq.com/news/2026/07/oracle-cloud-free-tier-limits/) on June 15, 2026. Too small. |
| Extra Google Cloud trials (yours or a friend's) | Against the trial's terms; the account can be shut down mid-render, and charges land on the friend's card. |
| Colab free tier | Usable for quick tests only: Google [publishes no limits](https://gpuperhour.com/blog/google-colab-alternatives) and idle sessions time out. |

## Recipes

**Chaining a Phantom run across 12-hour Kaggle sessions**

1. Upload the code and setup files as a private Kaggle dataset.
2. Each session: restart from the latest full dump, run until ~11 hours, write a full dump, copy it to `/kaggle/working`.
3. Download the render snapshots each session so the 20 GB limit only has to hold the latest restart dump.
4. Same pattern as your VM's resume scripts; `cloud_run.py --resume` already trims records after the checkpoint.

**Rendering the final on Kaggle**

1. Install the NVIDIA Vulkan driver files (as on the L4: `VK_ICD_FILENAMES` pointing at the NVIDIA ICD) or `mesa-vulkan-drivers` for lavapipe.
2. Give each session a frame range. Render it, encode those frames into a high-quality video chunk (H.264 CRF ~10, or lossless), and download it before the session ends.
3. At the end, join the chunks with ffmpeg's concat and do the final encode once.

**The 4K cost, worked out:** your 1080p final used 1920 × 1080 × 4 rays ≈ 8.3 M rays per frame. 4K at 1 ray per pixel is 3840 × 2160 ≈ 8.3 M rays: the same work. 2 rays per pixel doubles it. The 1080p final took ~13 hours on the L4, and your notes estimate 20–30 hours on the Mac. With 2 rays per pixel plus the nebula, expect ~60–90 Mac hours (my estimate). Keep separate photon-ring supersampling for the thin ring.

**Storage:** 2,592 4K frames as 16-bit PNGs could reach roughly 100 GB (my estimate). Save 8-bit PNGs or encode per chunk, and keep the source frames only until the chunk is checked.
