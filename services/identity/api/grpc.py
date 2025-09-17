"""Minimal gRPC surface for the identity service."""

from __future__ import annotations

from concurrent import futures
from datetime import datetime
from typing import Dict, Optional

import grpc
from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict, ParseDict

from ..service import (
    ApiKeyRotationRequiredError,
    ApiKeyValidationError,
    IdentityService,
    InactiveAccountError,
    InvalidCredentialsError,
    PasswordExpiredError,
    RefreshTokenError,
)

SERVICE_NAME = "identity.v1.Identity"


def _datetime_to_iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


class IdentityGrpcApi:
    """Implementation of gRPC handlers forwarding to :class:`IdentityService`."""

    def __init__(self, identity_service: IdentityService) -> None:
        self.identity_service = identity_service

    # ------------------------------------------------------------------
    # RPC method implementations
    # ------------------------------------------------------------------
    def issue_token(self, request: struct_pb2.Struct, context) -> struct_pb2.Struct:
        payload = MessageToDict(request, preserving_proto_field_name=True)
        try:
            token_pair = self.identity_service.issue_token(
                payload.get("tenant"),
                payload.get("username", ""),
                payload.get("password", ""),
                requested_scopes=payload.get("scope"),
                mfa_code=payload.get("mfa_code"),
            )
        except (InvalidCredentialsError, PasswordExpiredError) as exc:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, str(exc))
        except InactiveAccountError as exc:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
        response = {
            "access_token": token_pair.access_token,
            "refresh_token": token_pair.refresh_token,
            "expires_in": token_pair.expires_in,
            "expires_at": token_pair.expires_at.isoformat(),
            "issued_at": token_pair.issued_at.isoformat(),
            "scope": token_pair.scope,
            "roles": token_pair.roles,
            "username": token_pair.username,
            "tenant": token_pair.tenant,
            "password_expires_at": _datetime_to_iso(token_pair.password_expires_at),
        }
        return ParseDict(response, struct_pb2.Struct())

    def refresh_token(self, request: struct_pb2.Struct, context) -> struct_pb2.Struct:
        payload = MessageToDict(request, preserving_proto_field_name=True)
        scope_value = payload.get("scope")
        if isinstance(scope_value, str):
            scope_value = [part for part in scope_value.split() if part]
        elif isinstance(scope_value, list):
            scope_value = [str(part) for part in scope_value]
        else:
            scope_value = None
        try:
            token_pair = self.identity_service.refresh_token(
                payload.get("tenant"),
                payload.get("refresh_token", ""),
                scope=scope_value,
            )
        except RefreshTokenError as exc:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, str(exc))
        response = {
            "access_token": token_pair.access_token,
            "refresh_token": token_pair.refresh_token,
            "expires_in": token_pair.expires_in,
            "expires_at": token_pair.expires_at.isoformat(),
            "issued_at": token_pair.issued_at.isoformat(),
            "scope": token_pair.scope,
            "roles": token_pair.roles,
            "username": token_pair.username,
            "tenant": token_pair.tenant,
            "password_expires_at": _datetime_to_iso(token_pair.password_expires_at),
        }
        return ParseDict(response, struct_pb2.Struct())

    def introspect_token(self, request: struct_pb2.Struct, context) -> struct_pb2.Struct:
        payload = MessageToDict(request, preserving_proto_field_name=True)
        response = self.identity_service.introspect_token(
            payload.get("tenant"), payload.get("token", "")
        )
        result = {
            "active": response.active,
            "username": response.username,
            "subject": response.subject,
            "tenant": response.tenant,
            "scope": response.scope,
            "roles": response.roles,
            "issued_at": _datetime_to_iso(response.issued_at),
            "expires_at": _datetime_to_iso(response.expires_at),
            "token_type": response.token_type,
            "client_id": response.client_id,
            "claims": response.claims,
        }
        return ParseDict(result, struct_pb2.Struct())

    def rotate_password(self, request: struct_pb2.Struct, context) -> struct_pb2.Struct:
        payload = MessageToDict(request, preserving_proto_field_name=True)
        try:
            user = self.identity_service.rotate_password(
                payload.get("tenant"),
                payload.get("username", ""),
                payload.get("current_password", ""),
                payload.get("new_password", ""),
            )
        except InvalidCredentialsError as exc:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, str(exc))
        except InactiveAccountError as exc:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
        result = _identity_user_payload(user)
        return ParseDict(result, struct_pb2.Struct())

    def rotate_api_key(self, request: struct_pb2.Struct, context) -> struct_pb2.Struct:
        payload = MessageToDict(request, preserving_proto_field_name=True)
        try:
            api_key, user = self.identity_service.rotate_api_key(
                payload.get("tenant"),
                payload.get("username", ""),
                new_api_key=payload.get("new_api_key"),
            )
        except InactiveAccountError as exc:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
        result = _identity_user_payload(user)
        result["api_key"] = api_key
        return ParseDict(result, struct_pb2.Struct())

    def validate_api_key(self, request: struct_pb2.Struct, context) -> struct_pb2.Struct:
        payload = MessageToDict(request, preserving_proto_field_name=True)
        try:
            user = self.identity_service.validate_api_key(
                payload.get("tenant"), payload.get("api_key", "")
            )
        except ApiKeyRotationRequiredError as exc:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
        except ApiKeyValidationError as exc:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, str(exc))
        result = _identity_user_payload(user)
        return ParseDict(result, struct_pb2.Struct())

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------
    def as_generic_handler(self) -> grpc.GenericRpcHandler:
        return grpc.method_handlers_generic_handler(
            SERVICE_NAME,
            {
                "IssueToken": grpc.unary_unary_rpc_method_handler(
                    self.issue_token,
                    request_deserializer=struct_pb2.Struct.FromString,
                    response_serializer=struct_pb2.Struct.SerializeToString,
                ),
                "RefreshToken": grpc.unary_unary_rpc_method_handler(
                    self.refresh_token,
                    request_deserializer=struct_pb2.Struct.FromString,
                    response_serializer=struct_pb2.Struct.SerializeToString,
                ),
                "IntrospectToken": grpc.unary_unary_rpc_method_handler(
                    self.introspect_token,
                    request_deserializer=struct_pb2.Struct.FromString,
                    response_serializer=struct_pb2.Struct.SerializeToString,
                ),
                "RotatePassword": grpc.unary_unary_rpc_method_handler(
                    self.rotate_password,
                    request_deserializer=struct_pb2.Struct.FromString,
                    response_serializer=struct_pb2.Struct.SerializeToString,
                ),
                "RotateApiKey": grpc.unary_unary_rpc_method_handler(
                    self.rotate_api_key,
                    request_deserializer=struct_pb2.Struct.FromString,
                    response_serializer=struct_pb2.Struct.SerializeToString,
                ),
                "ValidateApiKey": grpc.unary_unary_rpc_method_handler(
                    self.validate_api_key,
                    request_deserializer=struct_pb2.Struct.FromString,
                    response_serializer=struct_pb2.Struct.SerializeToString,
                ),
            },
        )


def _identity_user_payload(user) -> Dict[str, object]:
    return {
        "username": user.username,
        "roles": list(user.roles),
        "tenant": user.tenant,
        "password_rotated_at": _datetime_to_iso(user.password_rotated_at),
        "password_expires_at": _datetime_to_iso(user.password_expires_at),
        "api_key_last_rotated_at": _datetime_to_iso(user.api_key_last_rotated_at),
        "api_key_expires_at": _datetime_to_iso(user.api_key_expires_at),
    }


def create_server(identity_service: IdentityService, *, max_workers: int = 10) -> grpc.Server:
    """Return a configured gRPC server instance."""

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    handler = IdentityGrpcApi(identity_service).as_generic_handler()
    server.add_generic_rpc_handlers((handler,))
    return server


__all__ = ["SERVICE_NAME", "IdentityGrpcApi", "create_server"]
