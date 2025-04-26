from fastapi.testclient import TestClient
from webapp.backend.app import app

client = TestClient(app)

def test_predict_endpoint():
    with open("tests/sample_image.jpg", "rb") as image_file:
        response = client.post("/predict", files={"file": ("sample_image.jpg", image_file, "image/jpeg")})
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) == 3
    for prediction in response.json()["predictions"]:
        assert "class" in prediction
        assert "score" in prediction
