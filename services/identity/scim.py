"""SCIM helper utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .models import RoleModel, UserModel
from .schemas import ScimEmail, ScimMeta, ScimName, ScimUser


@dataclass(slots=True)
class ScimUpdate:
    """Normalised SCIM attributes extracted from API requests."""

    username: Optional[str] = None
    display_name: Optional[str] = None
    email: Optional[str] = None
    active: Optional[bool] = None
    role_names: Optional[List[str]] = None
    attributes: Dict[str, object] = field(default_factory=dict)


def build_scim_user(user: UserModel, base_url: str) -> ScimUser:
    """Convert a :class:`UserModel` into a SCIM-compatible payload."""

    meta = ScimMeta(
        created=user.created_at,
        lastModified=user.updated_at,
        location=f"{base_url.rstrip('/')}/scim/v2/Users/{user.id}",
    )
    name_attrs = user.attributes.get("name") if isinstance(user.attributes, dict) else {}
    scim_name = None
    if name_attrs:
        scim_name = ScimName(
            givenName=name_attrs.get("givenName"),
            familyName=name_attrs.get("familyName"),
            formatted=name_attrs.get("formatted"),
        )
    emails: List[ScimEmail] = []
    if user.email:
        emails.append(ScimEmail(value=user.email, primary=True))
    roles = [
        {"value": role.name, "display": role.description or role.name}
        for role in user.roles
    ]
    return ScimUser(
        id=str(user.id),
        externalId=user.external_id,
        userName=user.username,
        active=user.is_active,
        emails=emails,
        displayName=user.display_name,
        roles=roles,
        name=scim_name,
        meta=meta,
    )


def parse_scim_payload(payload: Dict[str, object]) -> ScimUpdate:
    """Extract SCIM attributes for persistence."""

    update = ScimUpdate()
    update.username = str(payload.get("userName")) if payload.get("userName") else None
    update.display_name = payload.get("displayName") or None
    if "active" in payload:
        update.active = bool(payload["active"])

    name_payload = payload.get("name")
    if isinstance(name_payload, dict):
        update.attributes.setdefault("name", {})
        update.attributes["name"] = {
            "givenName": name_payload.get("givenName"),
            "familyName": name_payload.get("familyName"),
            "formatted": name_payload.get("formatted"),
        }

    emails_payload = payload.get("emails")
    if isinstance(emails_payload, list) and emails_payload:
        primary = None
        for entry in emails_payload:
            if not isinstance(entry, dict):
                continue
            if entry.get("primary"):
                primary = entry
                break
        primary = primary or emails_payload[0]
        update.email = primary.get("value") if isinstance(primary, dict) else None

    roles_payload = payload.get("roles")
    if isinstance(roles_payload, list):
        names = []
        for entry in roles_payload:
            if isinstance(entry, dict):
                value = entry.get("value") or entry.get("display")
                if value:
                    names.append(str(value))
            elif entry:
                names.append(str(entry))
        if names:
            update.role_names = names

    return update


def apply_scim_update(user: UserModel, update: ScimUpdate, roles: List[RoleModel]) -> None:
    """Apply the SCIM update to the SQLAlchemy model."""

    if update.username:
        user.username = update.username
    if update.display_name is not None:
        user.display_name = update.display_name
    if update.email is not None:
        user.email = update.email
    if update.active is not None:
        user.is_active = update.active
    if update.attributes:
        base_attrs = dict(user.attributes or {})
        base_attrs.update(update.attributes)
        user.attributes = base_attrs
    if update.role_names is not None:
        role_map = {role.name: role for role in roles}
        selected = [role_map[name] for name in update.role_names if name in role_map]
        user.roles = selected
    user.updated_at = datetime.utcnow()


__all__ = ["ScimUpdate", "apply_scim_update", "build_scim_user", "parse_scim_payload"]
