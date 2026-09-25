import inspect

from app.services.agenttools import vector_search


def test_vector_search_accepts_user_role():
    params = inspect.signature(vector_search).parameters
    assert "user_role" in params
    assert params["user_role"].default is None
