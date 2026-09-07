import sys
from types import SimpleNamespace
from unittest import TestCase, IsolatedAsyncioTestCase
from unittest.mock import Mock, patch, AsyncMock

from gliner.serve.config import GLiNERServeConfig
from gliner.serve.server import serve, _build_deployment


class TestHealthEndpoints(IsolatedAsyncioTestCase):
    def setUp(self):
        # Keep the real HTTP handler, but avoid starting Ray or loading a model.
        ray_serve = Mock()
        ray_serve.deployment.side_effect = lambda **kwargs: lambda cls: SimpleNamespace(
            bind=lambda config: cls.__new__(cls)
        )
        ray_serve.batch.side_effect = lambda **kwargs: lambda method: method
        with patch.dict(sys.modules, {"ray": SimpleNamespace(serve=ray_serve)}):
            self.deployment = _build_deployment(GLiNERServeConfig(model="test", route_prefix="/"))
        self.deployment.predict = AsyncMock(return_value={"entities": []})

    async def test_health_checks_need_no_body_or_inference(self):
        for path in ("/health", "/health-check", "/health/", "/health-check/"):
            with self.subTest(path=path):
                request = Mock(url=SimpleNamespace(path=path))
                request.json = AsyncMock(side_effect=ValueError("Health probes have no JSON body"))

                response = await self.deployment(request)

                self.assertEqual(response, {"status": "ok"})
                request.json.assert_not_awaited()
        self.deployment.predict.assert_not_awaited()

    async def test_prediction_still_works_at_gliner_path(self):
        request = Mock(url=SimpleNamespace(path="/gliner"))
        request.json = AsyncMock(return_value={"text": "John", "labels": ["person"]})

        response = await self.deployment(request)

        self.assertEqual(response, {"entities": []})
        self.deployment.predict.assert_awaited_once()
        self.assertEqual(self.deployment.predict.call_args.kwargs["text"], "John")
        self.assertEqual(self.deployment.predict.call_args.kwargs["labels"], ["person"])


class TestContainerServing(TestCase):
    def test_serving_binds_container_interface_and_root_route(self):
        config = GLiNERServeConfig(model="test", route_prefix="/", http_port=8080)
        ray = Mock()
        with patch.dict(sys.modules, {"ray": ray}), patch("gliner.serve.server._build_deployment") as build:
            serve(config)

        ray.serve.start.assert_called_once_with(
            detached=True, http_options={"host": "0.0.0.0", "port": 8080}
        )
        ray.serve.run.assert_called_once_with(build.return_value, name="gliner", route_prefix="/")
