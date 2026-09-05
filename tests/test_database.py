import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import datetime as dt
from database.database import session_scope, init_db
from database.models import Animal, AlertType, AlertSeverity
from database.crud import (
    get_or_create_animal,
    log_detection,
    log_tracking_event,
    log_behavior,
    create_alert,
    list_alerts,
    get_dashboard_counts,
)


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    init_db()


def test_get_or_create_animal():
    with session_scope() as db:
        uid = f"test_lion_{int(dt.datetime.now().timestamp())}"
        a1 = get_or_create_animal(db, track_uid=uid, species="lion")
        assert a1.id is not None
        assert a1.species == "lion"

        # Re-fetch should return existing
        a2 = get_or_create_animal(db, track_uid=uid, species="lion")
        assert a2.id == a1.id


def test_log_detection_and_tracking():
    with session_scope() as db:
        uid = f"test_zebra_{int(dt.datetime.now().timestamp())}"
        animal = get_or_create_animal(db, track_uid=uid, species="zebra")

        det = log_detection(
            db=db,
            animal_id=animal.id,
            species="zebra",
            confidence=0.91,
            bbox=(10, 20, 110, 120),
            camera_id="test_cam",
        )
        assert det.id is not None
        assert det.species == "zebra"

        trk = log_tracking_event(
            db=db,
            animal_id=animal.id,
            tracking_id=99,
            pos=(60, 70),
            speed_mps=2.4,
            distance_travelled_m=15.0,
            camera_id="test_cam",
        )
        assert trk.id is not None
        assert trk.speed_mps == 2.4


def test_alerts_and_counts():
    with session_scope() as db:
        alert = create_alert(
            db=db,
            alert_type=AlertType.FENCE_ESCAPE,
            severity=AlertSeverity.CRITICAL,
            message="Test fence breach alert",
            camera_id="test_cam",
        )
        assert alert.id is not None

        alerts = list_alerts(db, severity=AlertSeverity.CRITICAL)
        assert len(alerts) >= 1

        counts = get_dashboard_counts(db)
        assert counts["total_animals"] >= 1
