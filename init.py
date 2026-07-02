import nuke

# Register the gizmo node classes: the .gizmo files live in ./gizmos, which
# Nuke doesn't auto-scan just because the parent folder is on the plugin path.
# (Fixes "H2_SamViT: Unknown command".)
nuke.pluginAddPath("./gizmos")
