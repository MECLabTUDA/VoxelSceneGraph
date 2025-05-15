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

3. Run `scene-graph-prediction/configs/NeurIps2025/make_cfgs_rel.py` to generate the config files for all experiments from the template.
4. Before you can do relation prediction, you need to train the corresponding object detector. Check `object_detection_results.md` for more details.
5Run the experiment for each config file present in `scene-graph-prediction/configs/NeurIPS2025/rel_pred`. For instance:

```bash
sgpred_relation_train_net -c configs/NeurIps2025/rel_pred/all_rel_imp_use_gt1.yml
```
