"""Tests for API."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

class TestHealth:
    def test_health(self):
        assert client.get("/health").status_code == 200

class TestAutocomplete:
    def test_autocomplete(self):
        resp = client.post("/autocomplete", json={"text": "machine learning", "top_k": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["suggestions"]) > 0
        assert "context" in data
