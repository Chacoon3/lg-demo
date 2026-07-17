from __future__ import annotations

from pydantic import BaseModel

from lg_demo.utils.json_file_writer import JsonFileWriter


class FirstPayload(BaseModel):
    name: str
    count: int


class SecondPayload(BaseModel):
    value: str


def test_json_file_writer_appends_pretty_json_for_methods(tmp_path):
    writer = JsonFileWriter(tmp_path / "json-output")

    class Service:
        def __init__(self) -> None:
            self.calls = 0

        @writer
        def build(self, name: str) -> FirstPayload:
            self.calls += 1
            return FirstPayload(name=name, count=self.calls)

    service = Service()

    assert service.build("alpha") == FirstPayload(name="alpha", count=1)
    assert service.build("beta") == FirstPayload(name="beta", count=2)

    file_path = next((tmp_path / "json-output").glob("*.json"))
    contents = file_path.read_text(encoding="utf-8")

    assert '"name": "alpha"' in contents
    assert '\n  "count": 1\n' in contents
    assert contents.count('"name":') == 2
    assert "\n\n" in contents


def test_json_file_writer_uses_separate_files_per_class(tmp_path):
    writer = JsonFileWriter(tmp_path / "json-output")

    @writer
    def first() -> FirstPayload:
        return FirstPayload(name="first", count=1)

    @writer
    def second() -> SecondPayload:
        return SecondPayload(value="second")

    first()
    second()

    output_files = sorted((tmp_path / "json-output").glob("*.json"))

    assert len(output_files) == 2
    assert any("FirstPayload" in path.name for path in output_files)
    assert any("SecondPayload" in path.name for path in output_files)


def test_json_file_writer_ignores_non_model_returns(tmp_path):
    writer = JsonFileWriter(tmp_path / "json-output")

    @writer
    def plain_text() -> str:
        return "ok"

    assert plain_text() == "ok"
    assert list((tmp_path / "json-output").glob("*.json")) == []
