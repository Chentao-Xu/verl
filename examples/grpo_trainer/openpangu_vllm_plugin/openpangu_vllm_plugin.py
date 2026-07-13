"""vLLM plugin entry point for the OpenPangu checkpoint adapter."""

from importlib.util import find_spec


def register() -> None:
    if find_spec("pangu_infer") is None:
        return
    import verl.models.transformers.openpangu_vllm  # noqa: F401
