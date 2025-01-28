from theoden.security.auth import AuthenticationManager, UserRole

AuthenticationManager.create_yaml_and_users(
    "users.yaml",
    "topology.yaml",
    create_users=True,
    api_user="fedpath_guest",
    api_password='+Xx$j"H"N99M64dfN_(a=+l"vLBo?^',
    api_url="http://fedpath.gris.informatik.tu-darmstadt.de:15672/api/",
)
