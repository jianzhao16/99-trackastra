import torch
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks
from trackastra.data import example_data_bacteria

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load test data
imgs, masks = example_data_bacteria()        # imgs likely shaped (T, Y, X)

# Load a pretrained model
model = Trackastra.from_pretrained("general_2d", device=device)

# Track the cells
track_graph = model.track(imgs, masks, mode="greedy")  # or "ilp", "greedy_nodiv"

# Export to CTC + get tracked masks
ctc_tracks, masks_tracked = graph_to_ctc(track_graph, masks, outdir="tracked")

# Prepare napari layers
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
