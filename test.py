import torch
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks
from trackastra.data import example_data_bacteria

device = "cuda" if torch.cuda.is_available() else "cpu"

# load some test data images and masks
imgs, masks = example_data_bacteria()

# Load a pretrained model
model = Trackastra.from_pretrained("general_2d", device=device)

# or from a local folder
# model = Trackastra.from_folder('path/my_model_folder/', device=device)

# Track the cells
track_graph = model.track(imgs, masks, mode="greedy")  # or mode="ilp", or "greedy_nodiv"


# Write to cell tracking challenge format
ctc_tracks, masks_tracked = graph_to_ctc(
      track_graph,
      masks,
      outdir="tracked",
)

# Visualise in napari
napari_tracks, napari_tracks_graph, _ = graph_to_napari_tracks(track_graph)

import napari
viewer = napari.Viewer()
viewer.add_image(imgs, name="Images")                    # (T, Y, X)
viewer.add_labels(masks_tracked, name="Tracked Labels")  # (T, Y, X)
viewer.add_tracks(data=napari_tracks,
                  graph=napari_tracks_graph,
                  name="Tracks")

# --- Play the time axis at 10 seconds per frame ---
# Determine which axis is time; for (T, Y, X) it's axis 0.
# If you have axis labels, you could do:
# axis = viewer.dims.axis_labels.index('t') if 't' in viewer.dims.axis_labels else 0
#axis = 0

# Play at 0.1 FPS = 10 s/frame
#viewer.dims.play(axis=axis, fps=0.1, loop=True)

napari.run()