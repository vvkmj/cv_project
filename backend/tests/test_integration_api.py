import os
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database.session import SessionLocal, Base, engine

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield

def test_health_endpoints():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

    res_sys = client.get("/api/system/status")
    assert res_sys.status_code == 200
    assert "system" in res_sys.json()
    assert "workload" in res_sys.json()

    res_models = client.get("/api/models/status")
    assert res_models.status_code == 200
    data = res_models.json()
    assert "vehicle_detector" in data
    assert "helmet_detector" in data

def test_camera_crud_lifecycle():
    cam_id = "CAM-TEST-99"
    # Create
    new_cam = {
        "id": cam_id,
        "name": "Integration Test Camera",
        "location_label": "Test Boulevard 101",
        "expected_direction": {"angle_deg": 180.0, "dx": -1.0, "dy": 0.0},
        "passenger_limit": 2,
        "enabled_violations": {"no_helmet": True, "multiple_riders": True, "wrong_way": True}
    }
    create_res = client.post("/api/cameras", json=new_cam)
    assert create_res.status_code in [201, 400]

    # Read
    get_res = client.get(f"/api/cameras/{cam_id}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "Integration Test Camera"

    # Update
    patch_res = client.patch(f"/api/cameras/{cam_id}", json={"name": "Updated Camera Name", "passenger_limit": 1})
    assert patch_res.status_code == 200
    assert patch_res.json()["name"] == "Updated Camera Name"
    assert patch_res.json()["passenger_limit"] == 1

    # Delete
    del_res = client.delete(f"/api/cameras/{cam_id}")
    assert del_res.status_code == 204

    # Confirm 404
    get_after = client.get(f"/api/cameras/{cam_id}")
    assert get_after.status_code == 404

def test_media_upload_and_validation():
    # Unsupported file type test
    dummy_txt = b"Hello world text file"
    res_err = client.post(
        "/api/media/upload",
        files={"file": ("test.txt", dummy_txt, "text/plain")}
    )
    assert res_err.status_code == 400
    assert "Unsupported file format" in res_err.json()["detail"]

    # Valid image upload test
    dummy_jpg = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xFF\xDB"
    res_ok = client.post(
        "/api/media/upload",
        files={"file": ("test_frame.jpg", dummy_jpg, "image/jpeg")}
    )
    assert res_ok.status_code == 200
    data = res_ok.json()
    assert data["media_type"] == "image"
    assert "filename" in data

def test_image_analysis_pipeline():
    sample_img = "traffic_sample.jpg"
    res = client.post(
        "/api/analysis/image",
        data={"filename": sample_img, "camera_id": "CAM-01"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert data["media_type"] == "image"
    assert "summary" in data
    assert "violations" in data

def test_violations_listing_filtering_and_export():
    # Fetch list
    res = client.get("/api/violations?page=1&page_size=10")
    assert res.status_code == 200
    data = res.json()
    assert "total" in data
    assert "items" in data

    # Export CSV
    csv_res = client.get("/api/violations/export?format=csv")
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers["content-type"]
    assert "Violation ID" in csv_res.text

    # Export JSON
    json_res = client.get("/api/violations/export?format=json")
    assert json_res.status_code == 200
    assert isinstance(json_res.json(), list)

def test_violation_status_update():
    # Create or find an existing violation
    res = client.get("/api/violations?page=1&page_size=1")
    items = res.json()["items"]
    if items:
        viol_id = items[0]["id"]
        patch_res = client.patch(
            f"/api/violations/{viol_id}/status",
            json={"review_status": "confirmed"}
        )
        assert patch_res.status_code == 200
        assert patch_res.json()["review_status"] == "confirmed"

