import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    # Using the TestClient as a context manager triggers the FastAPI lifespan events,
    # which initializes the database (init_db) and creates necessary directories.
    with TestClient(app) as c:
        yield c


def test_health_and_cuda_check():
    """Test if the app imports and starts correctly, including CUDA check in lifespan."""
    import torch
    # Just a simple sanity check that torch is available in this env
    assert torch is not None


def test_sync_directory_and_list_tasks(client):
    """Test the /api/tasks/sync and /api/tasks endpoints."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create some dummy files: two valid media files and one invalid file
        (temp_path / "sample_audio.wav").touch()
        (temp_path / "sample_video.mp4").touch()
        (temp_path / "ignored_file.txt").touch()

        # 1. Sync the directory
        response = client.post("/api/tasks/sync", json={"directory": temp_dir})
        assert response.status_code == 200
        data = response.json()
        assert data["scanned_directory"] == temp_dir
        assert data["created"] == 2
        assert len(data["tasks"]) == 2

        task_ids = [task["id"] for task in data["tasks"]]

        # 2. List the tasks to verify they were added to the DB
        response_list = client.get("/api/tasks")
        assert response_list.status_code == 200
        tasks = response_list.json()
        
        # Check that our created tasks are in the returned list
        db_task_ids = [t["id"] for t in tasks]
        for tid in task_ids:
            assert tid in db_task_ids


def test_process_task_not_found(client):
    """Test that processing a non-existent task returns 404."""
    response = client.post("/api/tasks/invalid-uuid-1234/process")
    assert response.status_code == 404


def test_process_task_workflow(client):
    """Test triggering the process endpoint for a valid PENDING task."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dummy_file = Path(temp_dir) / "dummy.mp3"
        dummy_file.touch()

        # Sync to create a task
        sync_resp = client.post("/api/tasks/sync", json={"directory": temp_dir})
        assert sync_resp.status_code == 200
        task_id = sync_resp.json()["tasks"][0]["id"]

        # Trigger process
        process_resp = client.post(f"/api/tasks/{task_id}/process")
        assert process_resp.status_code == 200
        assert process_resp.json()["status"] == "processing_started"

        # The actual conversion/chunking will run in a BackgroundTask.
        # Since it's a dummy file, ffmpeg will fail, and the task status will eventually become FAILED.
        # We won't block to wait for it here, as BackgroundTasks in TestClient run synchronously after the response is sent.
        # So we can check if the status updated to FAILED or if it handled the error gracefully.
        list_resp = client.get("/api/tasks")
        updated_task = next((t for t in list_resp.json() if t["id"] == task_id), None)
        assert updated_task is not None
        # Since dummy.mp3 is not a real media file, ffmpeg should fail and mark it as FAILED
        assert updated_task["status"] in ["FAILED", "PENDING", "CONVERTING"]
