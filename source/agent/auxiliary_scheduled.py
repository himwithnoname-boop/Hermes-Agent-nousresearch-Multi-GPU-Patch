"""Streaming auxiliary provider seam; facade retains the public entrypoint."""

from agent.local_request_scheduler import guarded_provider


def relay_sync_stream(client, kwargs, *, provider=None, api_mode=None):
    from agent import auxiliary_client as aux

    from agent.auxiliary_wire import prepare_chat_messages, relay_chat_metadata

    kwargs = prepare_chat_messages(client, kwargs)
    # The bypass runs inside the provider callback, AFTER Relay has seen (and possibly
    # rewritten) the real conversation; applying it to `kwargs` would hand Relay an empty one.
    create = lambda request: client.chat.completions.create(**aux.bypass_chat_sdk_request_transform(request, client))
    create = guarded_provider(create)
    route = aux._relay_auxiliary_metadata(provider=provider, api_mode=api_mode)
    if route is None:
        return create(kwargs)
    provider_name, fallback_model, metadata = route
    from agent import relay_llm
    from agent.auxiliary_hooks import run_with_aux_hooks
    model_name = str(kwargs.get("model") or fallback_model)
    return run_with_aux_hooks(
        lambda: relay_llm.stream_current(
            kwargs, create, name=provider_name, model_name=model_name, finalizer=dict,
            metadata=relay_chat_metadata(metadata), completed_response_predicate=lambda value: hasattr(value, "choices"),
        ),
        aux_task=str(metadata.get("auxiliary_task") or ""), metadata=metadata, client=client, kwargs=kwargs,
        provider=provider_name, model=model_name, api_mode=str(metadata.get("api_mode") or ""), streaming=True,
    )
