import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from person_presence.config import PresenceConfig
from person_presence.event_store import EventStore
from person_presence.models import PersonDetection, PresenceEvent, WorldState
from person_presence.presence import PresenceManager
from person_presence.social import SocialAgent


class PresenceManagerTest(unittest.TestCase):
    def setUp(self):
        self.config = PresenceConfig(confirm_frames=2, exit_after_seconds=0.1)
        self.manager = PresenceManager(self.config)

    def detection(self, timestamp: str):
        return PersonDetection("camera_track_1", (10, 10, 110, 210), 0.9, timestamp)

    def test_confirmed_track_emits_enter_and_exit(self):
        now = datetime.now(timezone.utc)
        self.assertEqual(self.manager.update([self.detection(now.isoformat())], (640, 480)), [])
        entered = self.manager.update([self.detection((now + timedelta(milliseconds=20)).isoformat())], (640, 480))
        self.assertEqual([item.event_type for item in entered], ["person_entered"])
        self.assertEqual(self.manager.snapshot().active_count, 1)
        state = self.manager._tracks["camera_track_1"]
        state.last_seen_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        exited = self.manager.update([], (640, 480))
        self.assertEqual([item.event_type for item in exited], ["person_exited"])
        self.assertEqual(self.manager.snapshot().active_count, 0)

    def test_out_of_zone_detection_is_ignored(self):
        config = PresenceConfig(confirm_frames=1, zone=(0.0, 0.0, 0.2, 0.2))
        manager = PresenceManager(config)
        event = manager.update([self.detection(datetime.now(timezone.utc).isoformat())], (640, 480))
        self.assertEqual(event, [])
        self.assertEqual(manager.snapshot().active_count, 0)


class EventStoreTest(unittest.TestCase):
    def test_empty_store(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "events.sqlite3")
            self.assertEqual(store.recent(), [])
            self.assertEqual(store.today_entry_count(), 0)
            store.close()


class SocialAgentTest(unittest.TestCase):
    def test_only_first_entry_is_accepted_during_cooldown(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "events.sqlite3")
            agent = SocialAgent(store, lambda _: "你好，欢迎来到实验室。", cooldown_seconds=60)
            now = datetime.now(timezone.utc).isoformat()
            event = PresenceEvent("event_1", "person_entered", "camera_track_1", now, 0.9)
            world = WorldState(updated_at=now, persons=[], active_count=1)
            self.assertTrue(agent.submit(event, world))
            self.assertFalse(agent.submit(event, world))
            store.close()


if __name__ == "__main__":
    unittest.main()
