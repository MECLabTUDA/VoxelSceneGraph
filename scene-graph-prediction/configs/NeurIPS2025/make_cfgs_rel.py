from pathlib import Path


Path("rel_pred").mkdir(exist_ok=True)

for template_path in [
    "template_rel_imp_use_gt.yml",
    "template_rel_imp_use_pred.yml",
    "template_rel_motif_use_gt.yml",
    "template_rel_motif_use_pred.yml"
]:
    with open(template_path, "r") as f:
        template_str = "".join(f.readlines())

    for idx in range(1, 6):
        for name, ds in [
            ["BHSD", '("BHSD_rel",)'],
            ["CQ500", '("CQ500_rel",)'],
            ["HemSeg200", '("HemSeg200_rel",)'],
            ["INST", '("INSTANCE2022_rel",)'],
            ["all", '("BHSD_rel", "CQ500_rel", "HemSeg200_rel", "INSTANCE2022_rel")'],
        ]:
            current_cfg = (
                template_str
                .replace("{idx}", str(idx))
                .replace("{name}", name)
                .replace("{ds}", ds)
            )
            with open(f"rel_pred/{name}_{template_path[9:]}".replace('.yml', str(idx) + '.yml'), "w") as f:
                f.write(current_cfg)
