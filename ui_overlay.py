# ui_overlay.py - Viewport overlay rendering for interactive points and bbox
# Draws points, bounding box, and labels in Nuke's viewer for H2 SamViT

import nuke
import nuke.rotopaint as rp
from typing import List, Dict, Any, Optional, Tuple


class ViewerOverlay:
    """Manages viewport overlay rendering for H2 SamViT."""
    
    def __init__(self, node: nuke.Node):
        self.node = node
    
    def get_ui_params(self) -> Dict[str, Any]:
        """Get UI appearance parameters from the node."""
        ui_color_value = self.node.knob("ui_color").value()
        # AColor knob (type 19) returns [r, g, b, a]
        if isinstance(ui_color_value, (list, tuple)) and len(ui_color_value) >= 4:
            color = ui_color_value[:3]
            alpha = ui_color_value[3]
        elif isinstance(ui_color_value, (list, tuple)) and len(ui_color_value) == 3:
            color = ui_color_value
            alpha = 0.8
        else:
            color = [0, 0.652, 1]
            alpha = 0.8
        return {
            "show": self.node.knob("show_ui_overlays").value(),
            "show_labels": self.node.knob("show_point_labels").value(),
            "color": color,
            "alpha": alpha,
            "scale": self.node.knob("overlay_scale").value(),
        }
    
    def get_points(self) -> List[Dict[str, Any]]:
        """Get all enabled points."""
        from . import callbacks
        return callbacks.get_enabled_points(self.node)
    
    def get_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Get bounding box if enabled."""
        from . import callbacks
        return callbacks.get_bbox(self.node)


def draw_overlay(node: nuke.Node) -> str:
    """
    Generate TCL/Python code for drawing overlay in viewer.
    
    This is called by Nuke's viewer to render interactive elements.
    """
    overlay = ViewerOverlay(node)
    params = overlay.get_ui_params()
    
    if not params["show"]:
        return ""
    
    points = overlay.get_points()
    bbox = overlay.get_bbox()
    
    color = params["color"]
    alpha = params["alpha"]
    scale = params["scale"]
    
    # Build draw commands
    draw_cmds = []
    
    # Point radius based on scale
    point_radius = 6 * scale
    
    # Draw bounding box
    if bbox:
        x1, y1, x2, y2 = bbox
        draw_cmds.append(f"""
# Bounding box
glColor4f({color[0]}, {color[1]}, {color[2]}, {alpha})
glLineWidth({2 * scale})
glBegin(GL_LINE_LOOP)
glVertex2f({x1}, {y1})
glVertex2f({x2}, {y1})
glVertex2f({x2}, {y2})
glVertex2f({x1}, {y2})
glEnd()
        """)
        
        # Draw corner handles
        handle_size = 8 * scale
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for cx, cy in corners:
            draw_cmds.append(f"""
glBegin(GL_QUADS)
glVertex2f({cx - handle_size/2}, {cy - handle_size/2})
glVertex2f({cx + handle_size/2}, {cy - handle_size/2})
glVertex2f({cx + handle_size/2}, {cy + handle_size/2})
glVertex2f({cx - handle_size/2}, {cy + handle_size/2})
glEnd()
            """)
    
    # Draw points
    for p in points:
        x, y = p["x"], p["y"]
        idx = p["index"]
        is_fg = p["is_foreground"]
        
        # FG points: green, BG points: red
        if is_fg:
            pt_color = (0, 1, 0)  # Green
        else:
            pt_color = (1, 0, 0)  # Red
        
        # Draw filled circle
        draw_cmds.append(f"""
# Point {idx}
glColor4f({pt_color[0]}, {pt_color[1]}, {pt_color[2]}, {alpha})
glBegin(GL_TRIANGLE_FAN)
glVertex2f({x}, {y})
""")
        # Approximate circle with triangles
        import math
        for angle in range(0, 370, 30):
            rad = math.radians(angle)
            px = x + point_radius * math.cos(rad)
            py = y + point_radius * math.sin(rad)
            draw_cmds.append(f"glVertex2f({px}, {py})")
        
        draw_cmds.append("glEnd()")
        
        # Draw point border
        draw_cmds.append(f"""
glColor4f(1, 1, 1, {alpha})
glLineWidth({1.5 * scale})
glBegin(GL_LINE_LOOP)
""")
        for angle in range(0, 370, 30):
            rad = math.radians(angle)
            px = x + point_radius * math.cos(rad)
            py = y + point_radius * math.sin(rad)
            draw_cmds.append(f"glVertex2f({px}, {py})")
        
        draw_cmds.append("glEnd()")
        
        # Draw label
        if params["show_labels"]:
            label_y = y - point_radius - 12 * scale
            draw_cmds.append(f"""
# Label for point {idx}
glColor4f(1, 1, 1, {alpha})
drawText({x - 4 * scale}, {label_y}, "{idx}")
            """)
    
    return "\n".join(draw_cmds)


def register_overlay(node: nuke.Node):
    """Register the overlay drawing callback for a node."""
    # This would typically use Nuke's Python GL drawing API
    # or integrate with the node's draw() method
    pass


def create_interaction_handlers(node: nuke.Node):
    """
    Create mouse interaction handlers for the node.
    
    In a real implementation, this would use Nuke's viewer
    interaction API to handle clicks and drags.
    """
    pass


# Helper function for OpenGL text rendering (placeholder)
def draw_text_at(x: float, y: float, text: str, scale: float = 1.0):
    """Draw text at the specified position."""
    # In real implementation, would use Nuke's text rendering API
    pass
