from scene_graph_api.knowledge import RadiologyImageKG, ObjectClass, RelationRule, WhitelistFilter


knowledge = RadiologyImageKG(
    [
        ObjectClass(1, "Bleeding", has_mask=True, is_unique=False, is_ignored=False, color="#ffa602"),
        ObjectClass(2, "Ventricle", has_mask=True, is_unique=True,  is_ignored=False, color="#ff6362"),
        ObjectClass(3, "Middle", has_mask=True, is_unique=True,  is_ignored=False, color="#bc5090")
    ],
    [
        RelationRule(1, "causes shift of", WhitelistFilter([3]), WhitelistFilter([2])),
        RelationRule(2, "flows into", WhitelistFilter([3]), WhitelistFilter([1])),
        RelationRule(3, "causes asym. of", WhitelistFilter([3]), WhitelistFilter([1])),
    ],
)
knowledge.save("knowledge_graph.json")
