from app.security.roles import Role

# Very basic RBAC map defining what roles can access which endpoints/actions
ROLE_PERMISSIONS = {
    Role.ADMIN: ["*"],
    Role.OPERATOR: ["read:chat", "write:chat", "read:report", "write:report", "read:health"],
    Role.VIEWER: ["read:chat", "read:report", "read:health"]
}

def has_permission(role: Role, permission: str) -> bool:
    if role not in ROLE_PERMISSIONS:
        return False
    perms = ROLE_PERMISSIONS[role]
    if "*" in perms:
        return True
    return permission in perms
