import asyncio
import importlib

from matgraph import cognee_patches
from matgraph.schema import Material


class FakeEmbeddingEngine:
    def get_batch_size(self):
        return 100


class FakeVectorEngine:
    embedding_engine = FakeEmbeddingEngine()

    def __init__(self):
        self.indexed: dict[tuple[str, str], list[str]] = {}

    async def create_vector_index(self, type_name, field_name):
        pass

    async def index_data_points(self, type_name, field_name, batch):
        # What cognee embeds: the value of the point's (single) index field.
        self.indexed[(type_name, field_name)] = [
            getattr(dp, dp.metadata["index_fields"][0]) for dp in batch
        ]


def test_each_index_field_is_embedded_from_its_own_text():
    cognee_patches.apply()
    material = Material(material_id="mp-x", formula="LiCoO2", description="structure text", profile="profile text")
    engine = FakeVectorEngine()
    module = importlib.import_module("cognee.tasks.storage.add_data_points")
    asyncio.run(module.index_data_points([material], vector_engine=engine))
    assert engine.indexed[("Material", "formula")] == ["LiCoO2"]
    assert engine.indexed[("Material", "description")] == ["structure text"]
    assert engine.indexed[("Material", "profile")] == ["profile text"]
    # The original point's metadata is untouched.
    assert material.metadata["index_fields"] == ["formula", "description", "profile"]
