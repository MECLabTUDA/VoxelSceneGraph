from pathlib import Path


template_path = Path("template_detec_other.yaml")
with open(template_path, "r") as f:
    template_str = "".join(f.readlines())
Path("obj_detec").mkdir(exist_ok=True)

for idx in range(1, 6):
    for name, ds in [
        ["BHSD", '("BHSD",)'],
        ["CQ500", '("CQ500",)'],
        ["HemSeg200", '("HemSeg200",)'],
        ["INST", '("INSTANCE2022",)'],
        ["all", '("BHSD", "CQ500", "HemSeg200", "INSTANCE2022")'],
    ]:
        current_cfg = (
            template_str
            .replace("{idx}", str(idx))
            .replace("{name}", name)
            .replace("{ds}", ds)
        )
        with open(f"obj_detec/{name}_obj_detec{idx}.yaml", "w") as f:
            f.write(current_cfg)
