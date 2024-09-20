from scene_graph_api.knowledge import RadiologyImageKG, ObjectClass


knowledge = RadiologyImageKG(
    [
        ObjectClass(1, "Bleeding", has_mask=True, is_unique=False, is_ignored=False, color="#ffa602")
    ],
)
knowledge.save("knowledge_graph.json")
