"""GPU image surface for the Viewer — the "converter model" (napari / BigDataViewer).

A single textured quad is drawn per frame; each active channel's plane is uploaded to
a GPU texture **once** (cached by identity), and contrast (``lo/hi``), gamma, emission
colour and additive multi-channel compositing all happen in the **fragment shader** from
uniforms. Consequences that fix the "viewer updates way slower than the load" problem:

* Moving the T/Z slider only swaps which textures are bound (or uploads the new plane
  once) — no per-frame ``np.percentile``, no numpy→QImage→QPixmap copies.
* Dragging a contrast/colour/gamma control is a uniform change + ``update()`` — **zero**
  decode, zero re-upload.

Planes are uploaded as single-channel 32-bit float (``R32F``): the uint16→float cast is
done **once at upload** (not per frame), which lets the shader use a plain ``sampler2D``
for every channel — far more portable than mixing ``usampler2D``/``sampler2D`` banks, and
the normalisation is exact.

Overlays (points / labels / tracks) stay on the CPU: :class:`GLImageView` calls back into
the panel with a :class:`QPainter` after the GL draw, and exposes :meth:`plane_to_widget`
so the panel maps plane-space geometry onto the (pan/zoom) view.

If the GL context or shader is unusable this widget never crashes — it emits
:data:`gl_failed` once and the panel swaps in the CPU :class:`~nodelab_v2.viewer._ImageView`.

Reparenting the Viewer (docking it into the mini-map and back) makes Qt destroy and
recreate the context, which invalidates every texture/buffer/program made in the old
one. :meth:`GLImageView._release_gl` frees them while that context is still current and
:meth:`GLImageView.initializeGL` rebuilds on the new one, replaying ``_last_planes`` so
the image comes straight back instead of going black.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (
    QImage, QMatrix4x4, QPainter, QSurfaceFormat, QVector2D, QVector3D)
from PySide6.QtOpenGL import (
    QOpenGLBuffer, QOpenGLShaderProgram, QOpenGLTexture, QOpenGLVertexArrayObject)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

#: raw GL enums we need (QOpenGLWidget hands us a GL-ES-style functions object)
_GL_COLOR_BUFFER_BIT = 0x4000
_GL_TRIANGLE_STRIP = 0x0005
_GL_BLEND = 0x0BE2
_GL_FLOAT = 0x1406
_GL_TEXTURE_2D = 0x0DE1
_GL_TEXTURE0 = 0x84C0
_GL_TEXTURE_MIN_FILTER = 0x2801
_GL_TEXTURE_MAG_FILTER = 0x2800
_GL_TEXTURE_WRAP_S = 0x2802
_GL_TEXTURE_WRAP_T = 0x2803
_GL_NEAREST = 0x2600
_GL_LINEAR = 0x2601
_GL_CLAMP_TO_EDGE = 0x812F
_GL_R32F = 0x822E
_GL_R16 = 0x822A
_GL_RED = 0x1903
_GL_RGBA = 0x1908
_GL_RGBA8 = 0x8058
_GL_UNSIGNED_BYTE = 0x1401
_GL_UNSIGNED_SHORT = 0x1403
_GL_UNPACK_ALIGNMENT = 0x0CF5

_MAX_CH = 8                      # sampler bank size (microscopy rarely exceeds this)

#: PixelUnpackBuffer enum (PySide6 exposes it on the class or the nested Type enum)
_PIXEL_UNPACK = getattr(QOpenGLBuffer, "PixelUnpackBuffer", None)
if _PIXEL_UNPACK is None:            # pragma: no cover — older bindings nest it under Type
    _PIXEL_UNPACK = QOpenGLBuffer.Type.PixelUnpackBuffer

_VERT = """
#version 330 core
layout(location = 0) in vec2 a_pos;   // clip-space xy
layout(location = 1) in vec2 a_uv;    // texture uv
out vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

def _build_frag(n: int) -> str:
    # Sampler arrays MUST be indexed by a constant in GLSL 330 (dynamic indexing is
    # undefined pre-400 → links but samples black on many drivers), so unroll with a
    # guard per channel instead of a `for i` loop over u_tex[i]. Contrast is baked into
    # the 8-bit texture on the CPU (the float-texture upload path is broken in this
    # PySide6 build); the shader keeps colour + gamma as free uniforms.
    # The raw 16-bit value is packed into R (high byte) + G (low byte) of an RGBA8 texture
    # (only RGBA8/ubyte uploads reliably in this PySide6 build). The unpack + LUT window
    # are INLINED per channel with a constant sampler index — passing a sampler-array
    # element to a helper function returns a bad sampler on this driver. Windowing here in
    # the shader (u_vlo/u_vhi) makes contrast changes a free uniform update.
    # u_win[i] = vec2(vlo, vhi) — the LUT window is carried as a vec2 (set via QVector2D),
    # NOT two float uniforms: setUniformValue(loc, python_float) silently fails for a
    # float ARRAY element in this PySide6 build (vector overloads work), which was the
    # cause of the white frame. Windowing is inlined per channel with a constant index.
    # u_win[i] = vec3(vlo, vhi, gamma) — window + gamma carried as a vec3 (set via
    # QVector3D), NOT float array uniforms: setUniformValue(loc, python_float) silently
    # fails for a float ARRAY element in this PySide6 build (vector overloads work). The
    # transfer function is t = pow(clamp((v-vlo)/(vhi-vlo)), gamma) — gamma < 1 brightens
    # midtones, > 1 darkens. Unpack + windowing inlined per channel with a constant index
    # (passing a sampler-array element to a helper returns a bad sampler on this driver).
    lines = []
    for i in range(n):
        lines.append(
            f"    if (u_nchan > {i}) {{\n"
            f"        vec4 c{i} = texture(u_tex[{i}], v_uv);\n"
            f"        float v{i} = (c{i}.r * 65280.0 + c{i}.g * 255.0) / 65535.0;\n"
            f"        float t{i} = clamp((v{i} - u_win[{i}].x) / "
            f"max(u_win[{i}].y - u_win[{i}].x, 1e-6), 0.0, 1.0);\n"
            f"        rgb += pow(t{i}, max(u_win[{i}].z, 1e-3)) * u_color[{i}];\n"
            f"    }}")
    body = "\n".join(lines)
    return f"""
#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_tex[{n}];
uniform int   u_nchan;
uniform vec3  u_color[{n}];
uniform vec3  u_win[{n}];
void main() {{
    vec3 rgb = vec3(0.0);
{body}
    frag = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}}
"""


_FRAG = _build_frag(_MAX_CH)

#: diagnostic: output the interpolated UV as R,G so we can tell geometry/UV apart from
#: texture upload (enabled with NODELAB_GL_UVTEST=1).
_FRAG_UVTEST = """
#version 330 core
in vec2 v_uv;
out vec4 frag;
void main() { frag = vec4(v_uv, 0.0, 1.0); }
"""

#: diagnostic: output channel-0 texture red directly (no gamma/colour/nchan) — isolates
#: the texture upload from the compositing/uniform logic (NODELAB_GL_TEXTEST=1).
_FRAG_TEXTEST = """
#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_tex[%(N)d];
void main() { float v = texture(u_tex[0], v_uv).r; frag = vec4(v, v, v, 1.0); }
""" % {"N": _MAX_CH}


class GLImageView(QOpenGLWidget):
    """OpenGL textured-quad image surface with shader contrast/colour/compositing."""

    gl_failed = Signal()

    def __init__(self) -> None:
        super().__init__()
        import os
        self._debug = os.environ.get("NODELAB_GL_DEBUG", "") in ("1", "true", "yes")
        self._dbg_once = False
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)
        # per-channel GPU state
        self._tex: Dict[int, QOpenGLTexture] = {}
        self._tex_key: Dict[int, int] = {}     # channel → id(plane) currently uploaded
        self._range: Dict[int, Tuple[float, float]] = {}  # channel → data range mapped to [0,1]
        self._clim: Dict[int, Tuple[float, float]] = {}   # channel → LUT window (data units)
        self._color: Dict[int, Tuple[float, float, float]] = {}
        self._gamma: Dict[int, float] = {}
        self._active: List[int] = []
        # planes waiting to be uploaded — ALL GL work is deferred to paintGL (the only
        # place the context is guaranteed current on the right thread); uploading from
        # set_planes via makeCurrent() before the first paint segfaults on some drivers.
        self._pending: Dict[int, np.ndarray] = {}
        # the last planes handed to us, kept so a context REBUILD (Qt destroys and
        # recreates the context when the widget is reparented — docking the Viewer into
        # the mini-map and back) can re-upload them instead of showing a black frame.
        self._last_planes: Dict[int, np.ndarray] = {}
        self._img_wh: Optional[Tuple[int, int]] = None   # (w, h) of the texture
        # view transform (plane-space → widget): fit-scale × user zoom, + pan (widget px)
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._last_wh: Optional[Tuple[int, int]] = None
        self._panning: Optional[QPointF] = None
        # GL objects (built in initializeGL)
        self._prog: Optional[QOpenGLShaderProgram] = None
        self._vbo: Optional[QOpenGLBuffer] = None
        self._vao: Optional[QOpenGLVertexArrayObject] = None
        self._pbo: Optional[QOpenGLBuffer] = None
        self._ok = False
        self._failed = False
        #: the panel sets this to paint overlays after the GL draw: cb(painter)
        self.overlay_cb: Optional[Callable[[QPainter], None]] = None

    # ── GL lifecycle ───────────────────────────────────────────────────────────
    def initializeGL(self) -> None:
        import sys
        # Qt calls this again on a fresh context whenever the widget is reparented
        # (Viewer → mini-map → back). Every GL object built in the previous context is
        # dead by now, so start clean and re-queue the last planes for upload.
        self._tex.clear()
        self._tex_key.clear()
        self._prog = self._vbo = self._vao = None
        self._ok = False
        try:
            f = self.context().functions()
            f.glClearColor(0.02, 0.03, 0.05, 1.0)
            prog = QOpenGLShaderProgram(self)
            from PySide6.QtOpenGL import QOpenGLShader
            import os
            if os.environ.get("NODELAB_GL_UVTEST", "") in ("1", "true", "yes"):
                frag = _FRAG_UVTEST
            elif os.environ.get("NODELAB_GL_TEXTEST", "") in ("1", "true", "yes"):
                frag = _FRAG_TEXTEST
            else:
                frag = _FRAG
            if not prog.addShaderFromSourceCode(QOpenGLShader.Vertex, _VERT):
                raise RuntimeError("vertex: " + prog.log())
            if not prog.addShaderFromSourceCode(QOpenGLShader.Fragment, frag):
                raise RuntimeError("fragment: " + prog.log())
            if not prog.link():
                raise RuntimeError("link: " + prog.log())
            self._prog = prog
            vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            vbo.create()
            vbo.setUsagePattern(QOpenGLBuffer.DynamicDraw)
            self._vbo = vbo
            # A VAO is MANDATORY in a 3.3 core profile — without one bound, glDrawArrays
            # draws nothing (silently, no error) → a black frame.
            vao = QOpenGLVertexArrayObject(self)
            vao.create()
            self._vao = vao
            self._ok = True
            # tear our objects down while this context is still alive (a reparent
            # destroys it) and bring the picture back on the new one
            self.context().aboutToBeDestroyed.connect(self._release_gl)
            if self._last_planes:
                self._pending.update(self._last_planes)
            ver = self.context().format().version()
            print(f"[glview] GL ready — OpenGL {ver[0]}.{ver[1]}", file=sys.stderr, flush=True)
        except Exception as e:                   # noqa: BLE001 — degrade, never crash
            print(f"[glview] GL init failed → CPU fallback: {e}", file=sys.stderr, flush=True)
            self._ok = False
            if not self._failed:
                self._failed = True
                self.gl_failed.emit()

    def _release_gl(self) -> None:
        """Destroy this context's GL objects while it is still usable — Qt emits
        ``QOpenGLContext.aboutToBeDestroyed`` on a reparent (and at teardown). Nothing
        here may raise: the widget has to survive to be re-initialized on the next
        context, which :meth:`initializeGL` then rebuilds from ``_last_planes``."""
        try:
            self.makeCurrent()
            for tex in self._tex.values():
                try:
                    tex.destroy()
                except Exception:                # noqa: BLE001 — best-effort cleanup
                    pass
            for obj in (self._vbo, self._vao, self._pbo):
                if obj is not None:
                    try:
                        obj.destroy()
                    except Exception:            # noqa: BLE001
                        pass
            if self._prog is not None:
                self._prog.setParent(None)       # drop it now, with the context current
        except Exception:                        # noqa: BLE001 — never break teardown
            pass
        finally:
            self._tex.clear()
            self._tex_key.clear()
            self._prog = self._vbo = self._vao = self._pbo = None
            self._ok = False
            try:
                self.doneCurrent()
            except Exception:                    # noqa: BLE001
                pass

    def _fail(self) -> None:
        self._ok = False
        if not self._failed:
            self._failed = True
            self.gl_failed.emit()

    # ── public API (mirrors _ImageView / used by the panel) ─────────────────────
    def set_planes(self, planes: Dict[int, np.ndarray]) -> None:
        """Queue each channel's plane for upload and repaint. No GL here — the actual
        texture upload happens in :meth:`paintGL` (context guaranteed current). Refit is
        pure math, so it's safe to do now."""
        if not planes:
            return
        self._active = list(planes.keys())
        self._last_planes = dict(planes)      # replayed if the context is rebuilt
        wh = None
        for ch, plane in planes.items():
            h, w = plane.shape[:2]
            wh = (w, h)
            if self._tex_key.get(ch) != id(plane):
                self._pending[ch] = plane     # (re)upload only when the plane changed
        if wh is not None and wh != self._img_wh:
            self._img_wh = wh
            if self._last_wh != wh:                # new image size → refit (keep zoom on scrub)
                self._last_wh = wh
                self.fit()
        self.update()

    def _upload(self, ch: int, plane: np.ndarray) -> None:
        # Upload the RAW plane ONCE as a normalized 16-bit single-channel texture (R16).
        # The LUT window is applied in the shader (u_vlo/u_vhi) so contrast changes are
        # free — no re-upload, no re-decode. We record the data range that maps texel
        # [0,1] back to data units, so the viewer's clim (data units) → normalized window.
        a = np.asarray(plane)
        if a.dtype == np.uint16:
            u16 = a
            dmin, dmax = 0.0, 65535.0
        elif a.dtype == np.uint8:
            u16 = a.astype(np.uint16) * 257
            dmin, dmax = 0.0, 255.0
        else:
            af = np.nan_to_num(a.astype(np.float32))
            dmin, dmax = float(af.min()), float(af.max())
            if dmax <= dmin:
                dmax = dmin + 1.0
            u16 = ((af - dmin) / (dmax - dmin) * 65535.0).astype(np.uint16)
        u16 = np.ascontiguousarray(u16)
        h, w = u16.shape[:2]
        self._range[ch] = (dmin, dmax)
        if self._debug:
            import sys
            print(f"[glview] upload ch{ch}: shape={u16.shape} src_dtype={a.dtype} "
                  f"range=({dmin:.4g},{dmax:.4g}) clim={self._clim.get(ch)}",
                  file=sys.stderr, flush=True)
        # Pack the 16-bit value into R (high byte) + G (low byte) of an RGBA8 texture.
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = (u16 >> 8).astype(np.uint8)
        rgba[..., 1] = (u16 & 0xFF).astype(np.uint8)
        rgba[..., 3] = 255
        rgba = np.ascontiguousarray(rgba)
        f = self.context().functions()
        tex = self._tex.get(ch)
        if tex is None:
            tex = QOpenGLTexture(QOpenGLTexture.Target2D)
            tex.create()
            self._tex[ch] = tex
        f.glBindTexture(_GL_TEXTURE_2D, tex.textureId())
        f.glPixelStorei(_GL_UNPACK_ALIGNMENT, 1)
        # NEAREST: the packed high/low bytes must not be interpolated (that would corrupt
        # the reconstructed value). Display planes are pre-decimated, so this is fine.
        f.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_MIN_FILTER, _GL_NEAREST)
        f.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_MAG_FILTER, _GL_NEAREST)
        f.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_WRAP_S, _GL_CLAMP_TO_EDGE)
        f.glTexParameteri(_GL_TEXTURE_2D, _GL_TEXTURE_WRAP_T, _GL_CLAMP_TO_EDGE)
        f.glTexImage2D(_GL_TEXTURE_2D, 0, _GL_RGBA8, w, h, 0, _GL_RGBA,
                       _GL_UNSIGNED_BYTE, rgba.tobytes())
        self._tex_key[ch] = id(plane)

    def set_channel(self, ch: int, lo: float, hi: float,
                    color: Tuple[int, int, int], gamma: float = 1.0) -> None:
        """Set a channel's LUT window (``lo``/``hi``, data units) + emission colour. Both
        are free shader uniforms — no re-upload, no re-decode — so dragging a contrast
        slider is instantaneous. Call :meth:`refresh` to repaint."""
        self._clim[ch] = (float(lo), float(hi))
        self._color[ch] = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        self._gamma[ch] = float(gamma)

    def refresh(self) -> None:
        self.update()

    def clear(self) -> None:
        self._active = []
        self.update()

    # ── view transform ──────────────────────────────────────────────────────────
    def _fit_scale(self) -> float:
        if not self._img_wh:
            return 1.0
        w, h = self._img_wh
        if w <= 0 or h <= 0:
            return 1.0
        return min(self.width() / w, self.height() / h)

    def _disp_rect(self) -> Tuple[float, float, float, float]:
        """(origin_x, origin_y, disp_w, disp_h) of the image in widget pixels."""
        w, h = self._img_wh or (1, 1)
        s = self._fit_scale() * self._zoom
        dw, dh = w * s, h * s
        ox = (self.width() - dw) / 2.0 + self._pan.x()
        oy = (self.height() - dh) / 2.0 + self._pan.y()
        return ox, oy, dw, dh

    def plane_to_widget(self, px: float, py: float) -> QPointF:
        """Map a plane-space pixel (0..W, 0..H) to widget device-independent coords —
        the panel uses this to place overlay geometry on the pan/zoomed image."""
        w, h = self._img_wh or (1, 1)
        ox, oy, dw, dh = self._disp_rect()
        return QPointF(ox + (px / max(1, w)) * dw, oy + (py / max(1, h)) * dh)

    def fit(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def wheelEvent(self, e) -> None:
        f = 1.15 if e.angleDelta().y() > 0 else 1.0 / 1.15
        cursor = e.position()
        # keep the point under the cursor fixed while zooming
        before = self._img_at(cursor)
        self._zoom = max(0.05, min(80.0, self._zoom * f))
        after = self.plane_to_widget(*before) if before else None
        if before and after:
            self._pan += cursor - after
        self.update()

    def _img_at(self, wpt: QPointF):
        w, h = self._img_wh or (0, 0)
        if not w or not h:
            return None
        ox, oy, dw, dh = self._disp_rect()
        if dw <= 0 or dh <= 0:
            return None
        return ((wpt.x() - ox) / dw * w, (wpt.y() - oy) / dh * h)

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self._panning = e.position()

    def mouseMoveEvent(self, e) -> None:
        if self._panning is not None:
            self._pan += e.position() - self._panning
            self._panning = e.position()
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        self._panning = None

    def mouseDoubleClickEvent(self, e) -> None:
        self.fit()
        e.accept()

    # ── paint ────────────────────────────────────────────────────────────────────
    def paintGL(self) -> None:
        f = self.context().functions()
        f.glClear(_GL_COLOR_BUFFER_BIT)
        if self._ok and self._prog is not None:
            try:
                if self._pending:                # upload queued planes (context is current)
                    for ch, plane in list(self._pending.items()):
                        self._upload(ch, plane)
                    self._pending.clear()
                if self._active and self._img_wh:
                    self._draw_image(f)
            except Exception as e:               # noqa: BLE001
                import sys
                print(f"[glview] paint/upload failed → CPU fallback: {e}",
                      file=sys.stderr, flush=True)
                self._fail()
        # overlays (CPU QPainter) on top, sharing the same pan/zoom transform
        if self.overlay_cb is not None:
            try:
                p = QPainter(self)
                p.setRenderHint(QPainter.Antialiasing, True)
                self.overlay_cb(p)
                p.end()
            except Exception:                    # noqa: BLE001 — overlays are non-fatal
                pass

    def _draw_image(self, f) -> None:
        ox, oy, dw, dh = self._disp_rect()
        W, H = max(1, self.width()), max(1, self.height())

        def clip(wx, wy):
            return (wx / W) * 2.0 - 1.0, 1.0 - (wy / H) * 2.0
        # triangle strip: TL, BL, TR, BR ; uv has v flipped (image row 0 at top)
        corners = [(ox, oy, 0.0, 0.0), (ox, oy + dh, 0.0, 1.0),
                   (ox + dw, oy, 1.0, 0.0), (ox + dw, oy + dh, 1.0, 1.0)]
        verts = []
        for wx, wy, u, v in corners:
            cx, cy = clip(wx, wy)
            verts += [cx, cy, u, v]
        data = np.asarray(verts, dtype=np.float32).tobytes()

        prog, vbo, vao = self._prog, self._vbo, self._vao
        prog.bind()
        vao.bind()
        vbo.bind()
        vbo.allocate(data, len(data))
        prog.enableAttributeArray(0)
        prog.setAttributeBuffer(0, _GL_FLOAT, 0, 2, 16)
        prog.enableAttributeArray(1)
        prog.setAttributeBuffer(1, _GL_FLOAT, 8, 2, 16)

        active = self._active[:_MAX_CH]
        # Uniforms MUST go through uniformLocation() + the (location:int, value) overload:
        # PySide6 has no setUniformValue(name:str, scalar) overload (only location-based
        # for a single int/float), so name-based scalar calls raise.
        prog.setUniformValue(prog.uniformLocation("u_nchan"), int(len(active)))
        for i, ch in enumerate(active):
            r, g, b = self._color.get(ch, (1.0, 1.0, 1.0))
            dmin, dmax = self._range.get(ch, (0.0, 1.0))
            span = max(dmax - dmin, 1e-9)
            lo, hi = self._clim.get(ch, (dmin, dmax))
            vlo = (float(lo) - dmin) / span      # LUT window → texel-normalized [0,1]
            vhi = (float(hi) - dmin) / span
            gm = self._gamma.get(ch, 1.0)
            prog.setUniformValue(prog.uniformLocation(f"u_color[{i}]"), QVector3D(r, g, b))
            prog.setUniformValue(prog.uniformLocation(f"u_win[{i}]"),
                                 QVector3D(float(vlo), float(vhi), float(gm)))
            prog.setUniformValue(prog.uniformLocation(f"u_tex[{i}]"), int(i))
            tex = self._tex.get(ch)
            if tex is not None:
                f.glActiveTexture(_GL_TEXTURE0 + i)
                f.glBindTexture(_GL_TEXTURE_2D, tex.textureId())

        f.glDisable(_GL_BLEND)
        f.glDrawArrays(_GL_TRIANGLE_STRIP, 0, 4)

        f.glActiveTexture(_GL_TEXTURE0)
        prog.disableAttributeArray(0)
        prog.disableAttributeArray(1)
        vbo.release()
        vao.release()
        prog.release()


def default_surface_format() -> QSurfaceFormat:
    """A 3.3-core surface format — set as the app default BEFORE the QApplication so
    every QOpenGLWidget gets a modern context."""
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setVersion(3, 3)
    fmt.setSwapInterval(0)               # don't vsync-cap playback fps
    return fmt


def probe_gl_available() -> bool:
    """Whether the Viewer should try the GPU backend. We deliberately do **not** create a
    probe GL context here: on the headless ``offscreen``/``minimal`` Qt platforms that
    call crashes the process (a C++ segfault, uncatchable), and those are exactly the
    platforms used by the selftest probe and CI. So gate on the platform name — a real
    windowed session gets GL, headless stays on the CPU path — and let
    :meth:`GLImageView.initializeGL` handle a genuine driver failure at runtime by
    emitting :data:`GLImageView.gl_failed` (the panel then swaps to the CPU view)."""
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        name = (app.platformName() if app is not None else "").lower()
        return name not in ("", "offscreen", "minimal", "vnc")
    except Exception:                            # noqa: BLE001
        return False


__all__ = ["GLImageView", "default_surface_format", "probe_gl_available"]
