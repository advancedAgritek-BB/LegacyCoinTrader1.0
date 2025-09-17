"""Command line helpers for administering portfolio user accounts."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from dataclasses import asdict
from typing import Optional

from .config import PortfolioConfig
from .security import (
    IdentityError,
    IdentityService,
)


def _prompt_secret(prompt: str) -> str:
    try:
        return getpass.getpass(prompt)
    except Exception:
        return input(prompt)


def _resolve_password(value: Optional[str], prompt: str) -> str:
    if value:
        return value
    secret = _prompt_secret(prompt)
    if not secret:
        raise ValueError("A non-empty password is required")
    return secret


def _print_user(user) -> None:
    payload = asdict(user)
    print(json.dumps(payload, default=str, indent=2))


def create_user(args: argparse.Namespace) -> None:
    config = PortfolioConfig.from_env()
    service = IdentityService(config)
    password = _resolve_password(args.password, "Password: ")
    api_key = args.api_key or None
    user = service.create_user(
        args.username,
        password,
        args.role,
        api_key=api_key,
        is_active=not args.inactive,
    )
    _print_user(user)


def rotate_password(args: argparse.Namespace) -> None:
    config = PortfolioConfig.from_env()
    service = IdentityService(config)
    password = _resolve_password(args.password, "New password: ")
    if args.remove_api_key and args.api_key:
        raise ValueError("Provide --api-key or --remove-api-key, not both")
    user = service.rotate_password(
        args.username,
        password,
        api_key=args.api_key,
        remove_api_key=args.remove_api_key,
    )
    _print_user(user)


def set_api_key(args: argparse.Namespace) -> None:
    config = PortfolioConfig.from_env()
    service = IdentityService(config)
    if not args.clear and not args.api_key:
        raise ValueError("Provide --api-key or use --clear to remove it")
    api_key = args.api_key if not args.clear else None
    user = service.set_api_key(args.username, api_key)
    _print_user(user)


def set_active(args: argparse.Namespace) -> None:
    config = PortfolioConfig.from_env()
    service = IdentityService(config)
    user = service.set_active(args.username, not args.deactivate)
    _print_user(user)


def list_users(_args: argparse.Namespace) -> None:
    config = PortfolioConfig.from_env()
    service = IdentityService(config)
    users = service.list_users()
    for user in users:
        _print_user(user)


COMMANDS = {
    "create-user": create_user,
    "rotate-password": rotate_password,
    "set-api-key": set_api_key,
    "set-active": set_active,
    "list-users": list_users,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage hashed credentials stored in the portfolio database.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser(
        "create-user", help="Provision a new user with password rotation metadata."
    )
    create.add_argument("username", help="Unique username")
    create.add_argument("--role", default="viewer", help="Assigned role for RBAC enforcement")
    create.add_argument("--password", help="Password to set. Prompts securely when omitted.")
    create.add_argument("--api-key", help="Optional API key for service integrations")
    create.add_argument(
        "--inactive",
        action="store_true",
        help="Create the user in a disabled state for staged onboarding.",
    )

    rotate = subparsers.add_parser(
        "rotate-password",
        help="Rotate the password for an existing user and refresh the expiry timestamp.",
    )
    rotate.add_argument("username", help="Username to update")
    rotate.add_argument("--password", help="New password. Prompts when omitted.")
    rotate.add_argument("--api-key", help="Rotate the API key at the same time")
    rotate.add_argument(
        "--remove-api-key",
        action="store_true",
        help="Remove any existing API key assignment during rotation.",
    )

    api = subparsers.add_parser(
        "set-api-key", help="Assign or clear the API key associated with a user."
    )
    api.add_argument("username", help="Username to modify")
    api.add_argument("--api-key", help="API key to set. Required unless --clear is used.")
    api.add_argument(
        "--clear",
        action="store_true",
        help="Clear the stored API key without modifying the password.",
    )

    active = subparsers.add_parser(
        "set-active", help="Enable or disable a user account."
    )
    active.add_argument("username", help="Username to modify")
    active.add_argument(
        "--deactivate",
        action="store_true",
        help="Disable the user account instead of enabling it.",
    )

    subparsers.add_parser(
        "list-users",
        help="List every user along with audit metadata (password rotation, status).",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = COMMANDS.get(args.command)
    if command is None:
        parser.error(f"Unknown command: {args.command}")
    try:
        command(args)
    except IdentityError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
