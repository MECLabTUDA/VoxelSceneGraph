# Results with our Object Detector (scene-graph-prediction)

To reproduce our results, please follow these steps:
1. Install `scen-graph-prediction`
2. Download and prepare the data:
   - Go to our dataset page and download `BleedScene3D.zip` If you wish to evaluate on the PhysioNet dataset, you will need to separately download the image and normalize them as described on Kaggle.
   - Extract the archive in `scene-graph-prediction/datasets`. The folder structure should look like this:
```
scene-graph-prediction/
├─ configs/
│  ├─   ...
├─ datasets
│  ├─   BleedScene3D
│  ├─   ├─   normalized
├─ scene_graph_prediction
├─ tools
├─ ...
```

3. Run `scene-graph-prediction/configs/NeurIps2025/make_cfgs.py` to generate the config files for all experiments from the template.
4. Run the experiment for each config file present in `scene-graph-prediction/configs/NeurIPS2025/obj_detec`. For instance:

```bash
sgpred_detector_pretrain_net -c configs/NeurIps2025/obj_detec/all_obj_detec1.yaml
```
