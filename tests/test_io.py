import json
import os
import tempfile

from axiompy import Axiom


def test_save_load_axi():
    data = {"key": [1, 2, 3]}
    with tempfile.NamedTemporaryFile(suffix=".axi", delete=False) as f:
        path = f.name
    try:
        Axiom.io.save(data, path)
        loaded = Axiom.io.load(path)
        assert loaded == data
    finally:
        os.unlink(path)


def test_save_load_json():
    data = {"numbers": [1, 2, 3], "name": "test"}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        Axiom.io.save_json(data, path)
        loaded = Axiom.io.load_json(path)
        assert loaded == data
    finally:
        os.unlink(path)


def test_csv():
    rows = [["a", "b"], ["1", "2"]]
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        Axiom.io.save_csv(rows, path, headers=["x", "y"])
        loaded = Axiom.io.load(path)
        assert len(loaded) == 3  # header + 2 rows
    finally:
        os.unlink(path)


def test_txt():
    rows = [["Alice", "30"], ["Bob", "25"]]
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        path = f.name
    try:
        Axiom.io.save_txt(rows, path, headers=["Name", "Age"])
        lines = Axiom.io.load_txt(path)
        assert len(lines) >= 2
    finally:
        os.unlink(path)


def test_auto_detect():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        Axiom.io.save({"msg": "hello"}, path)
        loaded = Axiom.io.load(path)
        assert loaded == {"msg": "hello"}
    finally:
        os.unlink(path)
