from pathlib import Path

from candel.field import (
    field_allows_raw_product_reads,
    field_metadata,
    field_requires_cached_products,
    supported_field_names,
)
from candel.pvdata.field_products import los_field_cache_path


def test_raw_reads_are_allowed_for_cheap_fields():
    assert field_allows_raw_product_reads("Carrick2015")
    assert not field_requires_cached_products("Carrick2015")

    assert field_allows_raw_product_reads("ManticoreLocalCOLA")
    assert not field_requires_cached_products("ManticoreLocalCOLA")
    assert field_metadata("ManticoreLocalCOLA").cache_group == (
        "ManticoreLocalCOLA")
    assert field_metadata("ManticoreLocalCOLA").ngrid == 256
    assert field_metadata("ManticoreLocalCOLA").storage_schema == (
        "overdensity_velocity")


def test_non_cola_manticore_requires_cached_products():
    name = "ManticoreLocalSWIFT"

    assert field_requires_cached_products(name)
    assert not field_allows_raw_product_reads(name)
    assert field_metadata(name).cache_group == "ManticoreLocalSWIFT"
    assert field_metadata(name).ngrid == 1024
    assert field_metadata(name).production_method == "nbody_mas_sph"
    assert field_metadata(name).storage_schema == "density_momentum"


def test_unknown_fields_default_to_require_cached_products():
    name = "experimental_large_field"

    assert field_requires_cached_products(name)
    assert not field_allows_raw_product_reads(name)
    assert field_metadata(name).cache_group == "unknown"


def test_supported_field_names_include_loader_families():
    names = supported_field_names()

    assert "Carrick2015" in names
    assert "Lilow2024" in names
    assert "CF4" in names
    assert "CLONES" in names
    assert "HAMLET_V0" in names
    assert "HAMLET_V1" in names
    assert "CB1" in names
    assert "CB2" in names
    assert "ManticoreLocalCOLA" in names
    assert "ManticoreLocalSWIFT" in names


def test_old_manticore_names_are_not_supported():
    assert field_metadata("COLA_manticore_2MPP_MULTIBIN_N256_DES_V2").name == (
        "unknown")
    assert field_metadata("manticore_2MPP_MULTIBIN_N256_DES_V2").name == (
        "unknown")


def test_los_field_cache_path_uses_field_cache_dir(tmp_path):
    config = {"io": {"field_cache_dir": str(tmp_path)}}

    path = los_field_cache_path(
        config, "CF4", "ManticoreLocalCOLA", "data/los_<X>.hdf5",
        field_indices=0)

    path = Path(path)
    assert path.parent == tmp_path / "ManticoreLocalCOLA" / "los"
    assert path.name.startswith("los__CF4__field-0__")
    assert "v1" not in path.name
    assert "field-0" in path.name
    assert path.suffix == ".hdf5"
