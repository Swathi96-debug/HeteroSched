import json
import os
import subprocess
import sys
import unittest

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import server.app as server_app


class TestVercelAndScripts(unittest.TestCase):
    def setUp(self):
        server_app._load_initial_results()
        self.client = server_app.app.test_client()

    def test_vercel_json_configuration(self):
        vercel_path = os.path.join(WORKSPACE_ROOT, "vercel.json")
        self.assertTrue(os.path.exists(vercel_path), "vercel.json must exist")

        with open(vercel_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 1. Verify schema
        self.assertEqual(config.get("$schema"), "https://openapi.vercel.sh/vercel.json")

        # 2. Verify services: must contain frontend (vite) and server (flask), no spurious root app
        services = config.get("services", {})
        self.assertIn("frontend", services, "services must define 'frontend'")
        self.assertIn("server", services, "services must define 'server'")
        self.assertNotIn("app", services, "services must not define spurious root '.' app service")

        self.assertEqual(services["frontend"].get("root"), "frontend")
        self.assertEqual(services["frontend"].get("framework"), "vite")
        self.assertEqual(services["server"].get("root"), "server")
        self.assertEqual(services["server"].get("framework"), "flask")

        # 3. Verify rewrites: /api/(.*) -> server, /(.*) -> frontend
        rewrites = config.get("rewrites", [])
        self.assertGreaterEqual(len(rewrites), 2)

        api_rewrite = next((r for r in rewrites if r.get("source") == "/api/(.*)"), None)
        self.assertIsNotNone(api_rewrite, "Rewrite rule for /api/(.*) must exist")
        self.assertEqual(api_rewrite.get("destination", {}).get("service"), "server")

        spa_rewrite = next((r for r in rewrites if r.get("source") == "/(.*)"), None)
        self.assertIsNotNone(spa_rewrite, "Rewrite rule for /(.*) must exist")
        self.assertEqual(spa_rewrite.get("destination", {}).get("service"), "frontend")

    def test_run_all_help_flag(self):
        script_path = os.path.join(WORKSPACE_ROOT, "scripts", "run_all.sh")
        proc = subprocess.run([script_path, "--help"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--retrain", proc.stdout)

    def test_run_all_unknown_flag_rejected(self):
        script_path = os.path.join(WORKSPACE_ROOT, "scripts", "run_all.sh")
        proc = subprocess.run([script_path, "--bogus-flag-1234"], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Unknown argument", proc.stderr)

    def test_flask_import_from_root_and_server_dirs(self):
        # 1. Test importing from repository root
        proc_root = subprocess.run(
            [sys.executable, "-c", "import server.app; assert server.app.app is not None"],
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc_root.returncode, 0, f"Import from root failed: {proc_root.stderr}")

        # 2. Test importing from server/ directory (as Vercel does)
        proc_server = subprocess.run(
            [sys.executable, "-c", "import app; assert app.app is not None"],
            cwd=os.path.join(WORKSPACE_ROOT, "server"),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc_server.returncode, 0, f"Import from server/ failed: {proc_server.stderr}")

    def test_flask_routes_get(self):
        server_app._load_initial_results()

        # /api/system/info
        res_info = self.client.get("/api/system/info")
        self.assertEqual(res_info.status_code, 200)
        data_info = res_info.get_json()
        self.assertEqual(data_info.get("status"), "online")
        self.assertIn("gpu", data_info)

        # /api/traces
        res_traces = self.client.get("/api/traces")
        self.assertEqual(res_traces.status_code, 200)
        data_traces = res_traces.get_json()
        self.assertIn("mixed", data_traces.get("traces", {}))

        # /api/benchmark/status
        res_status = self.client.get("/api/benchmark/status")
        self.assertEqual(res_status.status_code, 200)

        # /api/benchmark/latest
        res_latest = self.client.get("/api/benchmark/latest")
        self.assertEqual(res_latest.status_code, 200)

        # /api/benchmark/download clean mode
        res_dl_clean = self.client.get("/api/benchmark/download?mode=clean")
        self.assertEqual(res_dl_clean.status_code, 200)
        self.assertEqual(res_dl_clean.mimetype, "text/csv")
        self.assertIn("attachment;", res_dl_clean.headers.get("Content-Disposition", ""))
        self.assertIn("task_id,type,priority", res_dl_clean.get_data(as_text=True))

        # /api/benchmark/download raw mode
        res_dl_raw = self.client.get("/api/benchmark/download?mode=raw")
        self.assertEqual(res_dl_raw.status_code, 200)
        self.assertEqual(res_dl_raw.mimetype, "text/csv")
        self.assertIn("attachment;", res_dl_raw.headers.get("Content-Disposition", ""))

    def test_flask_demo_sample_data_labeling(self):
        # Verify fallback demo labeling when root results.csv is absent (as on Vercel)
        orig_exists = os.path.exists
        try:
            os.path.exists = lambda p: False if (isinstance(p, str) and p.endswith("results.csv") and "sample" not in p) else orig_exists(p)
            server_app._load_initial_results()
            res = self.client.get("/api/benchmark/latest")
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertFalse(data.get("is_fresh"))
            self.assertEqual(data.get("source"), "demo_sample")
            self.assertEqual(data.get("run_id"), "sample_baseline")
            self.assertIn("Not Live GPU Measurement", data.get("trace_name", ""))
            self.assertEqual(len(data.get("tasks", [])), 200)
        finally:
            os.path.exists = orig_exists
            server_app._load_initial_results()

    def test_flask_benchmark_cloud_safety_missing_binary(self):
        import server.runner as runner
        orig_binary = runner.BINARY_PATH
        try:
            runner.BINARY_PATH = "/nonexistent/build/heterosched_bench"
            res = self.client.post("/api/benchmark/run", json={"trace": "mixed"})
            self.assertEqual(res.status_code, 503)
            data = res.get_json()
            self.assertIn("Real benchmark execution requires the local C++/CUDA runtime", data.get("error", ""))
            self.assertTrue(data.get("cloud_environment"))
        finally:
            runner.BINARY_PATH = orig_binary


if __name__ == "__main__":
    unittest.main()
