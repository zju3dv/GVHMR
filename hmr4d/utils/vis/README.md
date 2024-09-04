## Pytorch3D Renderer

Example:
```python
from hmr4d.utils.vis.renderer import Renderer
import imageio

fps = 30
focal_length = data["cam_int"][0][0, 0]
width, height = img_hw
faces = smplh[data["gender"]].bm.faces
renderer = Renderer(width, height, focal_length, "cuda", faces)
writer = imageio.get_writer("tmp_debug.mp4", fps=fps, mode="I", format="FFMPEG", macro_block_size=1)

for i in tqdm(range(length)):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img = renderer.render_mesh(smplh_out.vertices[i].cuda(), img)
    writer.append_data(img)
writer.close()
```