"""
validate.py — shared schema validator for all AIOI generators.

Usage:
    from validate import validate_records
    validate_records(records, "customer.schema.json")

Raises ValueError on first violation with record index and path.
"""
import json
from pathlib import Path
import jsonschema

_schema_cache: dict = {}
_SCHEMA_DIR = Path(__file__).parent.parent / "schemas"


def validate_records(records: list[dict], schema_name: str) -> None:
    """Validate a list of records against a named schema. Raises on first violation."""
    if schema_name not in _schema_cache:
        schema_path = _SCHEMA_DIR / schema_name
        _schema_cache[schema_name] = json.loads(schema_path.read_text())
    schema = _schema_cache[schema_name]
    for i, record in enumerate(records):
        try:
            jsonschema.validate(instance=record, schema=schema)
        except jsonschema.ValidationError as e:
            raise ValueError(
                f"Record {i} failed {schema_name} validation at "
                f"'{'/'.join(str(p) for p in e.absolute_path)}': {e.message}"
            ) from e


def validate_record(record: dict, schema_name: str) -> None:
    """Validate a single record. Raises ValueError on failure."""
    validate_records([record], schema_name)
