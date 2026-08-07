# @Author: Tristan Croll
# @Date:   21-Jul-2026
# @Email:  tcroll@altoslabs.com
# @Last modified by:   tcroll
# @Last modified time: 21-Jul-2026
# @License: Free for non-commercial use (see license.pdf)
# @Copyright: 2026 Tristan Croll
'''
Scratch/prototype validation section, "next rotamer"-style. One entry per
unparameterised residue in the selected model (labelled "<name>, chain <X>",
preceded by a small "edit in ChemSearch" button that opens the residue in the
ChimeraX-ChemSearch 2D structure editor -- optional sister bundle; the button is
disabled when it is not installed), each a row of: an arrow "cycle" button, a
grey "no imposed template" box, then one box per candidate MD template grouped
"by residue name" then "by topology similarity", ordered best-first. Template
boxes are coloured by RDKit FMCS overlap with the residue (green = identical,
purple-blue = poor); their tooltip carries the "possible templates" details with
that FMCS overlap.

Two outlines: a RED box marks the committed choice (applied to the model); a GREY
box marks a transient *preview*. A preview is a thin-stick copy of the template's
ideal structure, aligned onto the residue and superimposed -- low-commitment, it
does not touch the model. Interaction:
  * Hovering a box (>~2 frames, no button) previews it (grey) and flies the
    camera to the residue; leaving the box drops the preview. Red is unchanged.
  * Clicking a box moves the red box and rebuilds the residue to that template
    (the grey box clears the override).
  * Hovering the arrow flies the camera to the residue. Clicking it cycles the
    grey preview to the next box (wrapping) and reveals a green "accept" tick to
    its left: click the tick to commit the previewed box, or move the mouse off
    the whole row to reject (both the preview and the tick then disappear).

Context-aware grouping: residues that fail template matching only because they
are covalently modified are NOT shown one-by-one. They are clustered into the
units the parameterisation pipeline builds -- a covalent unit (e.g. a drug + the
cysteine it is bonded to), a metal coordination site, or a novel free ligand --
and each unit is a single row with one "Parameterise unit" button that runs that
pipeline (AM1-BCC) for the whole unit. This happens ONLY on the user's click; the
simulation build never parameterises on its own. A residue that merely needs
rebuilding to an EXISTING template still gets the candidate-box row described
above.

FMCS is computed lazily (only while the section is expanded) to keep the
background populate cheap on large models.
'''

from math import cos, pi

from Qt.QtWidgets import (
    QWidget,
    QLabel,
    QFrame,
    QScrollArea,
    QGridLayout,
    QSizePolicy,
    QApplication,
    QToolButton,
    QPushButton,
)
from Qt.QtCore import Qt, QTimer

from ..collapse_button import CollapsibleArea
from ..ui_base import UI_Panel_Base, DefaultVLayout, DefaultHLayout, busy_cursor

# matplotlib is already a hard ISOLDE dependency (see the Ramachandran plot), so
# sampling viridis for the score colour ramp adds no new requirement.
from matplotlib import colormaps

_VIRIDIS = colormaps['viridis']

BOX_SIZE = 20  # px; row controls are BOX_SIZE squares, and it sets the box height
# Suggestion boxes (candidate templates + the grey "no template" box) are half the
# control width but full height, so more candidates fit per line while staying
# aligned with -- and not shrinking -- the arrow/accept/edit buttons.
BOX_WIDTH = BOX_SIZE // 2
# Gap between the three box groups (no-template | name-match | topology-match).
# Within a group boxes are contiguous (row spacing 0); the gap only sits between
# groups. It no longer causes hover flicker -- crossing it stays inside the row,
# whose leaveEvent is what clears the preview (see BoxRow.leaveEvent).
GROUP_GAP = 14  # px
# Show roughly this many rows before the vertical scrollbar kicks in.
VISIBLE_ROWS = 8
# Gap (px) between the fixed-width name cell and the cycle arrow, within the single
# left-packed row that each entry is now composed into (see _compose_row).
GAP_AFTER_NAME = 6

# The lone "no imposed template" box is a flat grey (template boxes use viridis).
NO_TEMPLATE_COLOR = 'rgb(128, 128, 128)'
# Template boxes colour-code their RDKit FMCS overlap fraction (matched atoms /
# larger molecule) through a restricted slice of viridis -- 1.0 (identical) ->
# green, 0 (no shared substructure) -> purple-blue -- using a bit more of the low
# (bluer/purpler) end while still stopping short of viridis' purple/yellow ends.
VIRIDIS_LO = 0.1
VIRIDIS_HI = 0.7
# Candidate FMCS overlaps cluster high (they are, after all, candidates), so map
# [SCORE_FLOOR, 1.0] across the whole viridis band -- otherwise every real score
# lands in the all-green upper half and there is no visible gradient.
SCORE_FLOOR = 0.5
# Fill for a box whose FMCS overlap could not be computed (no CCD reference / a
# failed lookup). Deliberately a neutral grey, distinct both from the viridis
# ramp (so "unknown" never masquerades as "poor match") and from the flat grey
# of the "no imposed template" box.
UNCOMPUTED_COLOR = 'rgb(70, 72, 80)'
# Tooltip palette: a fixed dark background, distinct from every box fill, so a
# box's tooltip is never the same colour as the box itself.
TOOLTIP_BG = '#202020'
TOOLTIP_FG = '#f0f0f0'

# Camera fly-to (triggered by hovering a box). TEMPORARY: driven by frame count,
# not wall-clock, assuming ~45 fps so ~40 frames ~= 0.9 s (drifts with real frame
# rate). See navigate.ResidueStepper.step_to(easing=..., frames=...).
TRANSITION_FRAMES = 40
# Fallback centre-of-rotation-shift threshold (Angstroms): a hover-fly animates a
# move shorter than this and SNAPS (jump + reorient) a longer one. This is only
# the fallback when the ISOLDE setting is unavailable; the live, user-tweakable
# value is the 'preview_camera_snap_distance' setting (default in
# constants.defaults.CAMERA_SNAP_DISTANCE, exposed in ChimeraX Settings > ISOLDE),
# read per-fly in _fly_to_residue.
CAMERA_SNAP_DISTANCE = 50
# Hover dwell before a box previews / flies. Approximated in ms (>~2 frames)
# because a Qt-panel hover doesn't reliably tick the GL 'new frame' trigger.
HOVER_PREVIEW_DELAY_MS = 50
# The hover-fly skips only when the residue's framing atom (CA / C1' / centroid)
# is essentially AT the screen centre -- within this off-axis fraction (tan of
# the angle from the view axis). Kept tight (~0.03 ~= 1.7 deg) so a residue that
# is merely near-ish centre still flies; a real fly lands the atom on the axis
# (~0 deg), and rotating about the centre-of-rotation keeps it there, so this
# still suppresses re-rotation when you are already looking straight at it.
CAMERA_CENTERED_FRACTION = 0.03
# A listed residue is treated as "the one the camera is focused on" -- underlined,
# and the one whose preview is kept -- while its framing point is within this many
# Angstroms of the centre of rotation. Below the ~3.8 A CA-CA spacing so adjacent
# residues are still distinguished; a real navigation moves the CofR further.
COFR_MATCH_DISTANCE = 2.5
# FMCS is NP-hard; cap each residue-vs-template comparison so a large/symmetric
# residue can't stall the panel. RDKit returns its best match so far on timeout.
FMCS_TIMEOUT_S = 2
# Preview thin-stick appearance. (Carbons are painted with the model's own carbon
# colour at build time -- see NewSectionDialog._model_carbon_color.)
PREVIEW_STICK_RADIUS = 0.05

# --- "flatten to 2D" morph (pencil button) --------------------------------
# Clicking the pencil animates a residue's heavy atoms from their real 3D
# positions into a 2D chemical-diagram layout (computed by ChemSearch) laid on a
# plane facing the camera -- one view, no popup, and the atoms visibly resolve
# into the diagram so a newcomer keeps track of which atom is which. Driven by a
# QTimer; MORPH_STEP is the fraction of the transition added per tick, so the
# whole morph takes ~1/MORPH_STEP ticks * MORPH_TICK_MS end to end.
MORPH_TICK_MS = 16
MORPH_STEP = 1.0 / 24  # ~0.4 s at 16 ms/tick
# Scale applied to the (Angstrom-ish) 2D layout when embedding it in the scene.
# RDKit's ~1.5 A bond length already approx= scene scale, so 1.0 keeps the diagram
# the residue's size; a little larger spreads it out for legibility.
DEPICTION_SCALE = 1.4


def _ease_in_out_sine(t):
    '''Ease-in-out on a fraction t in [0, 1]: zero derivative (hence zero
    linear and angular camera velocity) at both ends, peak rate at the middle.'''
    return (1.0 - cos(pi * t)) / 2.0


def _framing_point(residue):
    '''The residue's framing point in SCENE coordinates -- the CA for an amino
    acid, C1' for a nucleotide, else the atom centroid -- matching what
    ResidueStepper centres on. None if the residue has no atoms.'''
    import numpy
    atoms = residue.atoms
    if not len(atoms):
        return None
    ref = residue.find_atom('CA') or residue.find_atom("C1'")
    pt = ref.scene_coord if ref is not None else atoms.scene_coords.mean(axis=0)
    return numpy.asarray(pt, dtype=float)


def _residue_is_centred(residue):
    '''True if the residue's framing point is already near the centre of the
    view: within CAMERA_CENTERED_FRACTION of the view axis, in front of the
    camera. The framing point matches what ResidueStepper actually centres on --
    the CA for an amino acid, C1' for a nucleotide, else the centroid -- NOT the
    mean of all atoms (whose offset from the CA would make a CA-centred residue
    read as off-centre and needlessly re-rotate). Orientation-independent, so a
    reorientation that keeps the residue framed is left undisturbed.'''
    import numpy
    atoms = residue.atoms
    if not len(atoms):
        return False
    ref = residue.find_atom('CA') or residue.find_atom("C1'")
    point = ref.scene_coord if ref is not None else atoms.scene_coords.mean(axis=0)
    cam = residue.structure.session.main_view.camera
    to_res = point - cam.position.origin()
    view_dir = cam.view_direction()
    depth = float(numpy.dot(to_res, view_dir))  # along the axis; >0 is in front
    if depth <= 0:
        return False
    perp = numpy.linalg.norm(to_res - depth * view_dir)  # distance off the axis
    return perp / depth < CAMERA_CENTERED_FRACTION


def _select_residue(residue):
    '''Make `residue` the current selection within its own model: deselect that
    model's atoms/bonds, then select the residue's atoms and their intra-residue
    bonds. Matches ISOLDE's own "focus an offending residue" idiom
    (isolde._handle_bad_template) and is scoped to the residue's structure, so a
    selection in any other open model is left untouched.'''
    m = residue.structure
    m.atoms.selected = False
    m.bonds.selected = False
    residue.atoms.selected = True
    residue.atoms.intra_bonds.selected = True


def _fly_to_residue(residue):
    '''Move the camera to view `residue` in its standard orientation (ISOLDE's
    ResidueStepper) AND make it the current selection (_select_residue). A move
    shorter than the configured fly-to distance (the 'preview_camera_snap_distance'
    ISOLDE setting) animates with an ease-in-out curve; a longer move snaps
    instantly (jump + reorient) rather than flying slowly across the model. The
    camera move is skipped when the residue is already centred (preserving a manual
    reorientation), but the selection still happens -- the residue was chosen, so
    it is selected whether or not the camera actually moves.'''
    if residue is None or residue.deleted:
        return
    _select_residue(residue)
    if _residue_is_centred(residue):
        return
    from ... import navigate, settings as _settings
    snap = CAMERA_SNAP_DISTANCE
    if _settings.basic_settings is not None:
        snap = _settings.basic_settings.preview_camera_snap_distance
    stepper = navigate.get_stepper(residue.structure)
    stepper.step_to(
        residue,
        easing=_ease_in_out_sine,
        frames=TRANSITION_FRAMES,
        max_interpolate_distance=snap
    )


def _fraction_to_viridis_css(fraction):
    '''CSS rgb() for an FMCS overlap fraction. None (uncomputable) -> a neutral
    grey. A real fraction is stretched from SCORE_FLOOR..1 across the restricted
    viridis band (green = identical, purple-blue = poor) so candidate scores --
    which bunch up near 1 -- actually spread into a visible gradient.'''
    if fraction is None:
        return UNCOMPUTED_COLOR
    norm = (max(0.0, min(1.0, fraction)) - SCORE_FLOOR) / (1.0 - SCORE_FLOOR)
    norm = max(0.0, min(1.0, norm))
    r, g, b, _ = _VIRIDIS(VIRIDIS_LO + norm * (VIRIDIS_HI - VIRIDIS_LO))
    return 'rgb({}, {}, {})'.format(int(r * 255), int(g * 255), int(b * 255))


def _tooltip_html(body):
    '''Wrap a tooltip body in a fixed dark, light-text table so the tooltip's
    colour is always distinct from the (viridis or grey) box it describes.'''
    return (
        '<table cellpadding="4" bgcolor="{bg}"><tr><td>'
        '<font color="{fg}">{body}</font></td></tr></table>'
    ).format(bg=TOOLTIP_BG, fg=TOOLTIP_FG, body=body)


def _match_tooltip(kind, tname, fraction, ccd_name, description):
    '''Rich-text tooltip for a template box: the match-kind heading, then the
    "possible templates" fields with the RDKit FMCS overlap as the score.'''
    frac_str = '{:.0%}'.format(fraction) if fraction is not None else 'n/a'
    lines = [
        '<b>{}</b>'.format(kind),
        'MD template: {}'.format(tname),
        'MCS overlap: {}'.format(frac_str),
        'CCD template: {}'.format(ccd_name),
    ]
    if description:
        lines.append(description)
    return _tooltip_html('<br>'.join(lines))


class SelectableBox(QFrame):
    '''A fixed-size coloured box (BOX_WIDTH x BOX_SIZE -- half-width, full height)
    for one candidate template (or the grey "no imposed template" box). Reports
    clicks/hovers to its BoxRow. Carries two independent outline states: committed
    (red) and previewed (grey); red wins.'''

    _COMMITTED_BORDER = '3px solid #e53935'  # red: the applied choice
    _PREVIEW_BORDER = '3px solid #9e9e9e'  # grey: the transient preview
    _PLAIN_BORDER = '1px solid rgba(0, 0, 0, 90)'

    def __init__(self, row, index, fill_css, tooltip, template_name, parent=None):
        super().__init__(parent)
        self.row = row
        self.index = index
        self.template_name = template_name  # None for the grey "no template" box
        self._fill_css = fill_css
        self.setFixedSize(BOX_WIDTH, BOX_SIZE)
        self._committed = False
        self._previewed = False
        self._hovered = False
        self._apply_style()
        self.setToolTip(tooltip)

    def set_committed(self, flag):
        if flag != self._committed:
            self._committed = flag
            self._apply_style()

    def set_previewed(self, flag):
        if flag != self._previewed:
            self._previewed = flag
            self._apply_style()

    def _apply_style(self):
        if self._committed:
            border = self._COMMITTED_BORDER
        elif self._previewed:
            border = self._PREVIEW_BORDER
        else:
            border = self._PLAIN_BORDER
        self.setStyleSheet(
            'border: {}; border-radius: 3px; background-color: {};'.format(
                border, self._fill_css
            )
        )

    def mousePressEvent(self, event):
        # A click commits: red moves here and the model is rebuilt to this box.
        if event.button() == Qt.MouseButton.LeftButton:
            self.row.click(self.index)
            event.accept()
            return
        super().mousePressEvent(event)

    def enterEvent(self, event):
        # Track hover synchronously (a flag is reliable on the very first entry,
        # where underMouse() can still read False when the deferred check fires),
        # then preview/fly after a short dwell; see _maybe_preview.
        self._hovered = True
        if QApplication.mouseButtons() == Qt.MouseButton.NoButton:
            QTimer.singleShot(HOVER_PREVIEW_DELAY_MS, self._maybe_preview)
        super().enterEvent(event)

    def leaveEvent(self, event):
        # Only drop the dwell flag here; the transient preview is cleared at the
        # ROW level (BoxRow.leaveEvent), so moving between boxes -- which stays
        # inside the row -- never blanks the preview (no flicker).
        self._hovered = False
        super().leaveEvent(event)

    def _maybe_preview(self):
        # Deferred hover: fly/preview only if still hovered with no button held.
        # The hovered flag (set in enterEvent) is more reliable than underMouse()
        # on the first entry. Guarded in case the box/row was deleted meanwhile.
        if not self._hovered or QApplication.mouseButtons() != Qt.MouseButton.NoButton:
            return
        try:
            self.row.hover_preview(self.index)
        except RuntimeError:
            pass


class ArrowButton(QToolButton):
    '''The per-row "cycle" button. Hovering it flies the camera to this residue;
    clicking advances the grey preview to the next box (wrapping) and reveals the
    accept button (green tick) to its left.'''

    def __init__(self, row, parent=None):
        super().__init__(parent)
        self.row = row
        self._hovered = False
        self.setArrowType(Qt.RightArrow)
        self.setAutoRaise(True)
        self.setFixedSize(BOX_SIZE, BOX_SIZE)
        self.clicked.connect(self._clicked)

    def _clicked(self, *_):
        self.row.arrow_clicked()

    def enterEvent(self, event):
        # Hovering the arrow flies the camera to this residue, after a short dwell
        # so a quick pass-over on the way to a box doesn't trigger a fly.
        self._hovered = True
        QTimer.singleShot(HOVER_PREVIEW_DELAY_MS, self._maybe_fly)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        super().leaveEvent(event)

    def _maybe_fly(self):
        if self._hovered:
            try:
                self.row.hover_arrow()
            except RuntimeError:
                pass


class AcceptButton(QPushButton):
    '''Appears to the LEFT of the arrow once the arrow is clicked: a green tick
    that commits the previewed template. Hidden otherwise, but keeps its layout
    slot (retainSizeWhenHidden) so the row never shifts when it appears/hides.'''

    def __init__(self, row, parent=None):
        super().__init__('✓', parent)  # check mark
        self.row = row
        self.setFixedSize(BOX_SIZE, BOX_SIZE)
        self.setToolTip('Accept this template')
        self.setStyleSheet(
            'QPushButton { background-color: #2e7d32; color: white; '
            'font-weight: bold; border: 1px solid #1b5e20; border-radius: 3px; }'
            'QPushButton:hover { background-color: #43a047; }'
        )
        sp = self.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.setSizePolicy(sp)
        self.setVisible(False)
        self.clicked.connect(self._clicked)

    def _clicked(self, *_):
        self.row.accept_clicked()


class ChemSearchButton(QToolButton):
    '''A compact button (leftmost in a row, just before the residue label) that
    opens this residue in the ChimeraX-ChemSearch 2D structure editor via
    seed_from_residue. ChemSearch is an OPTIONAL sister bundle -- ISOLDE only
    ports its conversion code, it does not depend on it -- so when the bundle is
    not installed the button is disabled with an explanatory tooltip rather than
    hidden (more discoverable than a silently absent control).'''

    def __init__(self, dialog, residue, available=True, parent=None):
        super().__init__(parent)
        self._dialog = dialog
        self.residue = residue  # held across time -- check .deleted before use
        self.setText('✎')  # pencil: "draw / edit this structure"
        self.setAutoRaise(True)  # flat until hovered, matching the arrow button
        self.setFixedSize(BOX_SIZE, BOX_SIZE)
        if available:
            self.setToolTip(
                'Flatten this residue to a 2D diagram in place '
                '(click again to restore; Shift-click to open the '
                'full ChemSearch editor)'
            )
            self.clicked.connect(self._clicked)
        else:
            self.setEnabled(False)
            self.setToolTip('ChimeraX-ChemSearch is not installed')

    def _clicked(self, *_):
        # Left-click flattens the residue to a 2D diagram in the 3D view (the
        # default -- one view, no popup); Shift-click still opens the full
        # ChemSearch 2D editor for actual editing.
        from Qt.QtWidgets import QApplication
        from Qt.QtCore import Qt
        shift = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)
        self._dialog._pencil_clicked(self.residue, open_editor=shift)


class ResidueNameLabel(QLabel):
    '''The "<name>, chain <X>" label at the start of a row. Clicking it flies the
    camera to the residue -- so the whole line, not just the small cycle arrow, is
    a fly-to target (the arrow still flies on hover). A pointing-hand cursor hints
    that it is clickable. The label is UNDERLINED while its residue is the one the
    camera is currently focused on (driven by NewSectionDialog._on_frame_drawn).'''

    def __init__(self, text, residue, parent=None):
        super().__init__(text, parent)
        self.residue = residue  # held across time -- _fly_to_residue checks .deleted
        self._focused = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_focused(self, flag):
        # Underline iff this residue is currently centred in the view. Coerce to a
        # native bool: _residue_is_centred yields a numpy.bool_, which PyQt6's
        # setUnderline rejects.
        flag = bool(flag)
        if flag == self._focused:
            return
        self._focused = flag
        f = self.font()
        f.setUnderline(flag)
        self.setFont(f)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            _fly_to_residue(self.residue)
            event.accept()
            return
        super().mousePressEvent(event)


class BoxRow(QWidget):
    '''One residue's row: an accept button (hidden until the arrow is clicked),
    the cycle arrow, the grey "no template" box, then the viridis template boxes.
    Owns the committed (red) and preview (grey) indices and drives the dialog's
    preview / commit for this residue.'''

    def __init__(self, name_cands, comp_cands, residue=None, dialog=None, parent=None):
        super().__init__(parent)
        # Held across time -- always check `.deleted` before use.
        self.residue = residue
        self._dialog = dialog
        self._label = None  # sibling residue label widget (set via set_label)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        hl = DefaultHLayout()
        hl.setSpacing(0)  # contiguous boxes -> no dead pixels to flicker over
        # Accept button (green tick) to the LEFT of the arrow -- hidden until the
        # arrow is clicked, but keeps its slot so the row never shifts.
        self._accept_btn = AcceptButton(self)
        hl.addWidget(self._accept_btn)
        # Cycle arrow, between the accept button and the first (grey) box.
        hl.addWidget(ArrowButton(self))
        self.boxes = []
        idx = 0
        # Far-left grey "no imposed template" box (committed by default).
        self.boxes.append(
            SelectableBox(self, idx, NO_TEMPLATE_COLOR, _tooltip_html('untemplated'), None)
        )
        hl.addWidget(self.boxes[-1])
        idx += 1
        for cands in (name_cands, comp_cands):
            if cands:
                hl.addSpacing(GROUP_GAP)
            for c in cands:
                self.boxes.append(
                    SelectableBox(self, idx, c['fill'], c['tooltip'], c['template_name'])
                )
                hl.addWidget(self.boxes[-1])
                idx += 1
        hl.addStretch()
        self.setLayout(hl)
        # Grey box committed by default (matches the model's "no override" state);
        # nothing is applied until the user acts.
        self._committed_index = 0
        self._preview_index = None
        self._arrow_armed = False
        self._refresh_outlines()

    def set_label(self, label):
        # Sibling residue label, kept for reference. It no longer drives any
        # leave-based rejection: previews now persist until the camera's centre of
        # rotation moves off the residue (see NewSectionDialog._on_frame_drawn).
        self._label = label

    # --- outline / armed state -------------------------------------------
    def _refresh_outlines(self):
        for i, b in enumerate(self.boxes):
            b.set_committed(i == self._committed_index)
            b.set_previewed(i == self._preview_index)

    def _set_armed(self, flag):
        # "Armed" == the arrow was clicked and the accept button is showing; the
        # accept button's visibility tracks this flag exactly.
        self._arrow_armed = flag
        self._accept_btn.setVisible(flag)

    def _set_preview(self, index):
        self._preview_index = index
        self._refresh_outlines()
        if self._dialog is None:
            return
        if index is None:
            self._dialog.remove_preview(self)
        else:
            self._dialog.show_preview(self, self.boxes[index].template_name)

    def clear_preview_outline(self):
        # Called by the dialog when another row takes over the single preview.
        self._preview_index = None
        self._set_armed(False)
        self._refresh_outlines()

    def _commit(self, index):
        self._committed_index = index
        self._refresh_outlines()
        if self._dialog is not None and self.residue is not None \
                and not self.residue.deleted:
            self._dialog.apply_template(self.residue, self.boxes[index].template_name)

    # --- box hover -------------------------------------------------------
    def hover_preview(self, index):
        # Hover dwell on a box: preview it (grey) and fly the camera. Red is
        # untouched. A box hover ends any arrow-cycle (hides the accept button).
        self._set_armed(False)
        self._set_preview(index)
        _fly_to_residue(self.residue)

    def hover_arrow(self):
        # Hovering the arrow just flies the camera to this residue; it does not
        # touch the preview or the committed / armed state.
        _fly_to_residue(self.residue)

    # Note: there is deliberately no leaveEvent-based rejection. A preview (and its
    # armed accept button) persists when the mouse leaves the row; it is dropped
    # only when the camera's centre of rotation moves off the residue, or replaced
    # when another row is hovered (NewSectionDialog._on_frame_drawn / show_preview).

    # --- box click -------------------------------------------------------
    def click(self, index):
        # Commit: red moves here, model is rebuilt; any preview / arm is dropped.
        self._set_armed(False)
        self._set_preview(None)
        self._commit(index)

    # --- arrow / accept --------------------------------------------------
    def arrow_clicked(self):
        # Advance the grey preview to the next box, wrapping, and reveal the accept
        # button. Red (committed) is untouched.
        start = self._committed_index if self._preview_index is None \
            else self._preview_index
        self._set_armed(True)
        self._set_preview((start + 1) % len(self.boxes))

    def accept_clicked(self):
        # Clicking the accept tick commits the previewed box.
        if self._preview_index is not None:
            index = self._preview_index
            self._set_armed(False)
            self._set_preview(None)
            self._commit(index)


class _NameScroll(QScrollArea):
    '''A fixed-width, scrollbar-less viewport for a residue-name label, so a long
    combined-unit name ("08J 1 (Z) + CYS 145 (A)") can't stretch the name column.
    No scrollbar is ever shown; the name can still be panned with a horizontal (or
    shift+) wheel, while a plain vertical wheel is passed through so the residue
    list still scrolls under the cursor. The wrapped label keeps its click-to-fly
    and focus-underline behaviour.'''

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.setWidget(label)
        self.setWidgetResizable(False)  # keep the label at its full natural width
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedHeight(label.sizeHint().height())
        # Transparent so it reads as a plain label, not an inset boxed widget.
        self.setStyleSheet('QScrollArea { background: transparent; border: none; }')
        self.viewport().setStyleSheet('background: transparent;')

    def wheelEvent(self, event):
        # Pan the name only on a horizontal (or shift-modified) wheel; let a plain
        # vertical wheel propagate so the enclosing residue list still scrolls.
        dx = event.angleDelta().x()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            dx = dx or event.angleDelta().y()
        if dx:
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - dx)
            event.accept()
        else:
            event.ignore()


class ParameteriseRow(QWidget):
    '''Row for a unit that needs a *fresh* MD template built -- a covalent unit, a
    metal site, or a novel free ligand with no existing template to rebuild to. A
    single button runs the existing parameterisation pipeline (AM1-BCC) for the
    whole unit. The button is disabled, with an explanatory note, when the unit
    cannot be built here: too large for AM1-BCC, an unsupported metal (no bundled
    LJ parameters, e.g. Mo), or a metal site whose donors could not be resolved.'''

    def __init__(self, descriptor, dialog=None, parent=None):
        super().__init__(parent)
        self._descriptor = descriptor
        self._dialog = dialog
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        hl = DefaultHLayout()
        kind = descriptor['kind']
        btn = QPushButton('Parameterise ligand' if kind == 'free' else 'Parameterise unit')
        # Reasons the unit can't be built here (button disabled + note shown):
        # an unsupported metal, an unresolved metal site, or too large for AM1-BCC.
        note = descriptor.get('unsupported') or descriptor.get('error')
        if note is None and descriptor['too_big']:
            note = (
                '{} heavy atoms -- too large for AM1-BCC; parameterise externally'.format(
                    descriptor['num_heavy_atoms']
                )
            )
        if note is not None:
            btn.setEnabled(False)
            btn.setToolTip(note)
        else:
            btn.clicked.connect(self._clicked)
        hl.addWidget(btn)
        if note is not None:
            lbl = QLabel(note)
            lbl.setStyleSheet('color: #b0b0b0; font-style: italic;')
            hl.addSpacing(6)
            hl.addWidget(lbl)
        hl.addStretch()
        self.setLayout(hl)

    def _clicked(self, *_):
        if self._dialog is not None:
            self._dialog.parameterise_unit(self._descriptor)


class NewSectionPanel(CollapsibleArea):

    def __init__(self, session, isolde, parent, gui, **kwargs):
        super().__init__(gui, parent, title="New section", **kwargs)
        cd = self.content = NewSectionDialog(session, isolde, gui, self)
        self.setContentLayout(cd.main_layout)


class NewSectionDialog(UI_Panel_Base):
    '''One entry per unparameterised residue; see the module docstring.'''

    def __init__(self, session, isolde, gui, collapse_area, sim_sensitive=False):
        super().__init__(
            session, isolde, gui, collapse_area.content_area, sim_sensitive=sim_sensitive
        )
        self.container = collapse_area
        mf = self.main_frame
        ml = self.main_layout = DefaultVLayout()

        scroll = self._scroll = QScrollArea(mf)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        inner = QWidget()
        grid = self._grid = QGridLayout(inner)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        inner.setLayout(grid)
        scroll.setWidget(inner)
        ml.addWidget(scroll)
        self._size_scroll(0)

        self.rows = []
        self._dirty = False
        # Set True by cleanup() so the self-rescheduling _process_next build chain
        # stops if the panel is torn down mid-populate (e.g. the window is closed).
        self._deleted = False
        self._preview_row = None
        self._preview_structure = None
        # Active "flatten to 2D" morph (pencil button), or None: a dict of the
        # throwaway depiction structure, its atom collection + per-atom
        # start/target coords, colour table, missing-atom mask, animation clock and
        # QTimer. Mutually exclusive with a template preview (each stops the other);
        # torn down by every panel exit path via _stop_morph.
        self._morph = None
        # While a preview is shown we hide the residue it stands in for, remembering
        # it + its atoms' display state so leaving restores exactly that. (The
        # preview's link to its chain neighbours is drawn as stub atoms/bonds inside
        # the preview structure itself -- see _build_preview -- so it needs no
        # separate lifecycle here.)
        self._hidden_residue = None
        self._saved_displays = None
        # Row name-labels (ResidueNameLabel), so _on_frame_drawn can underline the
        # one whose residue sits at the centre of rotation. Rebuilt each populate.
        self._name_labels = []
        # Scene-coord framing point of the residue whose preview is currently shown;
        # the preview is dropped once the centre of rotation moves away from it --
        # but only after it has first ARRIVED there (_preview_settled), so the
        # fly-in to a freshly-shown preview doesn't immediately discard it.
        self._preview_center = None
        self._preview_settled = False
        # Cheap change-detection: the focus/preview-drop check only recomputes when
        # the centre of rotation actually moved (its bytes differ).
        self._last_cofr_key = None
        self._frame_handler = self.session.triggers.add_handler(
            'frame drawn', self._on_frame_drawn
        )
        # Whether the optional ChimeraX-ChemSearch bundle is importable; probed
        # once, lazily, by _chemsearch_available (None => not yet probed).
        self._chemsearch_avail = None
        # Progressive population: rows are built one residue per event-loop turn
        # so they appear as they arrive. _build_gen invalidates an in-flight
        # incremental build if the panel is refreshed again mid-stream.
        self._build_gen = 0
        self._pending = []
        # Memoised CCD-derived lookups, keyed by ccd_name, so each component is
        # fetched at most once (the CCD network-fetch fallback is the main source
        # of both the populate delay and the "Error fetching CCD ..." log spam).
        self._tmpl_mol_cache = {}  # ccd_name -> RDKit mol or None
        self._preview_buildable = {}  # ccd_name -> bool (False = don't retry)
        # Recompute on the unparameterised-residue trigger (correct whether the
        # section is open or closed), but do the expensive FMCS/build only when
        # visible: a trigger while collapsed just marks the panel dirty.
        self._isolde_trigger_handlers.append(
            isolde.triggers.add_handler(isolde.UNPARAMETERISED_RESIDUE, self._refresh)
        )
        self.container.expanded.connect(self._refresh)

    def cleanup(self):
        # Stop the self-rescheduling _process_next build chain before our Qt
        # widgets are destroyed, then let the base class drop trigger handlers.
        self._deleted = True
        self._stop_morph()
        if self._frame_handler is not None:
            self.session.triggers.remove_handler(self._frame_handler)
            self._frame_handler = None
        super().cleanup()

    def _refresh(self, *_):
        if self._deleted:
            return
        if self.container.is_collapsed:
            self._dirty = True
            return
        self._dirty = False
        self._clear_rows()
        self._build_gen += 1
        # Persistent "Scan" button at the top (grid row 0): it re-runs detection and
        # adds hydrogens first on click (matching needs them). It never disappears
        # -- clicking again just rescans -- so detected rows populate BENEATH it. It
        # is appended to self.rows, which indexes the grid rows, so those rows start
        # on the next line rather than overwriting the button.
        self._add_scan_row()
        # Hydrogens are a prerequisite: template matching gates on an element
        # signature that COUNTS hydrogens, so an unprotonated model reads as if
        # every residue were unparameterised. We don't auto-add them (a side
        # effect); the user presses Scan. With no model / no hydrogens yet, the
        # panel is just the button.
        m = self.isolde.selected_model
        if m is None or not self._model_has_hydrogens(m):
            self._size_scroll(1)
            return
        # Cheap bulk step: which residues are unparameterised + their raw
        # candidate sources. The expensive per-candidate FMCS is deferred to
        # _process_next so rows appear progressively rather than all at once.
        self._pending = self._detect()
        self._size_scroll(len(self._pending) + 1)
        self._process_next(self._build_gen)

    @staticmethod
    def _model_has_hydrogens(model):
        '''Whether `model` contains any hydrogen atoms. Used to gate the section:
        an unprotonated model can't be meaningfully checked for unparameterised
        residues (template matching counts H).'''
        import numpy
        return bool((model.atoms.element_numbers == 1).any())

    def _add_scan_row(self):
        '''The persistent "Scan for unparameterized residues" button at grid row 0.
        Clicking it adds hydrogens (matching needs them) then repopulates; it stays
        put so a re-click simply rescans. Appended to self.rows so the detected rows
        beneath it start on the next grid line (len(self.rows)).'''
        btn = QPushButton('Scan for unparameterized residues  [will add hydrogens]')
        btn.setStyleSheet('QPushButton { font-weight: bold; padding: 4px 10px; }')
        btn.setToolTip(
            'Add hydrogens if needed (ISOLDE needs a fully protonated model, and '
            'template matching counts hydrogens), then list the residues with no '
            'matching MD template. Safe to click again to rescan.'
        )
        btn.clicked.connect(lambda *_: self._scan())
        self._grid.addWidget(btn, 0, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)
        self.rows.append(btn)

    def _scan(self):
        '''Scan-button action: add hydrogens (ISOLDE's addh convention), then
        repopulate. Runs only on the user's explicit click -- the "[will add
        hydrogens]" label flags that side effect -- and is safe to repeat. Always
        refreshes at the end so the Scan button (and any results) are redrawn.'''
        m = self.isolde.selected_model
        if m is not None and not m.deleted:
            try:
                from chimerax.atomic import AtomicStructures
                from chimerax.addh import cmd as addh_cmd
                with busy_cursor(self.session, 'Adding hydrogens...'):
                    addh_cmd.cmd_addh(self.session, AtomicStructures([m]), hbond=True)
            except Exception as e:
                self.session.logger.warning(
                    'New section: could not add hydrogens ({}: {})'.format(
                        e.__class__.__name__, e
                    )
                )
        with busy_cursor(self.session, 'Scanning for unparameterized residues...'):
            self._refresh()

    def _process_next(self, gen):
        # Build one entry's row per event-loop turn (so each paints as it lands),
        # bailing out if a newer refresh has superseded this build, or the panel
        # was torn down (window closed) while this deferred build was queued. An
        # entry is either a covalent/metal UNIT (a single "Parameterise unit" row)
        # or one free residue (the existing candidate-box / rebuild row, or -- when
        # it has no viable existing template -- a "Parameterise ligand" row).
        if self._deleted or gen != self._build_gen or not self._pending:
            return
        entry = self._pending.pop(0)
        i = len(self.rows)
        try:
            if entry[0] == 'unit':
                descriptor = entry[1]
                self._add_parameterise_row(i, descriptor, self._unit_label(descriptor))
            else:
                _, residue, kind, payload, descriptor = entry
                name_cands, comp_cands = self._candidates_for(kind, payload, residue)
                label = '{}, chain {}'.format(residue.name, residue.chain_id)
                if descriptor is not None and not name_cands and not comp_cands:
                    # A novel free ligand with no existing-template candidate: offer
                    # to build a fresh template rather than show an empty box row.
                    self._add_parameterise_row(i, descriptor, label)
                else:
                    cell, label_w = self._label_cell(label, residue)
                    row = BoxRow(name_cands, comp_cands, residue=residue, dialog=self)
                    row.set_label(label_w)
                    line = self._compose_row(cell, row)
                    self._grid.addWidget(line, i, 0, 1, 2)
                    self.rows.append(line)
        except RuntimeError:
            # A Qt widget (e.g. the grid) was destroyed while this build was queued
            # -- the panel is gone; stop the chain rather than crash.
            self._deleted = True
            return
        if self._pending:
            QTimer.singleShot(0, lambda: self._process_next(gen))

    def _compose_row(self, cell, box_row=None):
        '''Pack a row's column-0 label `cell` and its `box_row` (the cycle arrow +
        suggestion boxes; None for the blank parameterise rows) into a SINGLE
        left-hugging line, added spanning both grid columns. Doing the packing here
        -- instead of leaning on the two-column grid, which sized each column to fit
        and then centred the fixed-width cell / box row within it, opening gaps on
        both sides -- keeps the pencil + name flush left and the arrow just after
        the name. The leading pieces are fixed width, so arrows still line up across
        rows.'''
        line = QWidget()
        hl = DefaultHLayout()
        hl.setSpacing(0)
        hl.addWidget(cell)
        if box_row is not None:
            hl.addSpacing(GAP_AFTER_NAME)
            hl.addWidget(box_row)
        hl.addStretch()
        line.setLayout(hl)
        return line

    def _add_parameterise_row(self, i, descriptor, label):
        '''A label for a unit / novel free ligand at grid row `i`, with NO action
        widget beside it. The "Parameterise unit/ligand" action is temporarily
        disabled (it is broken), so rather than a dead button the row is just the
        label. (ParameteriseRow and parameterise_unit are kept for when the action
        is restored.) The ChemSearch edit button targets the unit's seed. Still
        appends to self.rows so the grid row counter (len(self.rows)) stays right.'''
        cell, _label_w = self._label_cell(label, descriptor['seed'])
        line = self._compose_row(cell)
        self._grid.addWidget(line, i, 0, 1, 2)
        self.rows.append(line)

    def _label_cell(self, label_text, residue):
        '''Column-0 cell for a row: a compact "edit in ChemSearch" button followed
        by the residue label. Returns (cell_widget, label_widget) -- the label is
        returned separately because a BoxRow watches its leave events (set_label).
        `residue` is the row's representative residue (a free ligand, or a unit's
        seed) handed to the ChemSearch editor.'''
        cell = QWidget()
        hl = DefaultHLayout()
        hl.setSpacing(4)
        hl.addWidget(ChemSearchButton(self, residue, available=self._chemsearch_available()))
        label_w = ResidueNameLabel(label_text, residue)
        label_w.setToolTip(label_text)  # full name on hover (the column is narrow)
        self._name_labels.append(label_w)  # for the camera-focus underline
        # Cap the name column so a long combined-unit label ("08J 1 (Z) + CYS 145
        # (A)") can't stretch it: the label lives in a fixed-width, scrollbar-less
        # viewport ~30% wider than a short "ABC + DE" name. A longer name is clipped
        # but still reachable -- hover for the full text (tooltip), or pan it with a
        # horizontal / shift wheel (no scrollbar; a plain vertical wheel still
        # scrolls the list -- see _NameScroll).
        fm = label_w.fontMetrics()
        name_w = int((fm.horizontalAdvance('ABC + DE') + 8) * 1.3)
        sa = _NameScroll(label_w)
        sa.setFixedWidth(name_w)
        hl.addWidget(sa)
        # A "..." in an always-reserved slot, shown only when the name is wider than
        # the viewport -- flags the overflow without making the column width vary
        # from row to row.
        dots = QLabel('…' if fm.horizontalAdvance(label_text) > name_w else '')
        dots.setFixedWidth(fm.horizontalAdvance('…') + 2)
        dots.setToolTip(label_text)
        hl.addWidget(dots)
        cell.setLayout(hl)
        # Fixed width so column 0 stays tight and uniform: an expanding cell (the old
        # trailing stretch) let the column soak up space and pushed the column-1
        # arrow far to the right regardless of how narrow the name viewport was.
        cell.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        return cell, label_w

    def _chemsearch_available(self):
        '''Whether the optional ChimeraX-ChemSearch bundle is importable (probed
        once and cached). ISOLDE does not depend on ChemSearch -- it only ports its
        conversion code -- so the per-row editor button is enabled only when the
        bundle is actually installed. find_spec does not execute the module, so the
        probe is cheap and side-effect-free.'''
        if self._chemsearch_avail is None:
            import importlib.util
            self._chemsearch_avail = (
                importlib.util.find_spec('chimerax.chemsearch') is not None
            )
        return self._chemsearch_avail

    @staticmethod
    def _unit_label(descriptor):
        '''Label for a unit row -- its member residues, seed first, e.g.
        "08J 1 (Z) + CYS 145 (A)".'''
        seed = descriptor['seed']
        members = [seed] + [r for r in descriptor['residues'] if r is not seed]
        parts = [
            '{} {} ({})'.format(r.name, r.number, r.chain_id) for r in members if not r.deleted
        ]
        return ' + '.join(parts) if parts else '(deleted residue)'

    def _size_scroll(self, n_rows):
        # Size the scroll area to its content, up to VISIBLE_ROWS (then scroll).
        # Fixes the default that showed only ~3 rows regardless of content.
        stride = BOX_SIZE + self._grid.verticalSpacing()
        h = max(1, min(n_rows, VISIBLE_ROWS)) * stride + 8
        self._scroll.setFixedHeight(h)

    def _clear_rows(self):
        self.remove_preview()
        grid = self._grid
        while grid.count():
            w = grid.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self.rows = []
        self._name_labels = []

    # --- preview management (one superimposed preview at a time) ----------
    def show_preview(self, row, template_name):
        # A template preview and a flatten-to-2D morph are mutually exclusive: a
        # hover preview taking over cancels any flattened depiction.
        self._stop_morph()
        # A different row taking over clears the previous row's grey outline.
        if self._preview_row is not None and self._preview_row is not row:
            self._preview_row.clear_preview_outline()
        self._delete_preview_structure()
        self._preview_row = row
        if template_name is not None:
            self._preview_structure = self._build_preview(row.residue, template_name)
            if self._preview_structure is not None:
                # Hide the whole residue so the preview stands in for it cleanly --
                # no model atoms (templated or not) showing under/around it. The
                # residue's own bonds to its neighbours hide with it, so the chain
                # link is redrawn as stub atoms/bonds in _build_preview.
                self._hide_replaced(row.residue)
                # Remember where this residue sits so _on_frame_drawn can drop the
                # preview once the centre of rotation moves off it. Seed "settled"
                # from the CURRENT CofR: if we are already centred here (the fly
                # will be skipped, so the CofR won't change to re-arm it) the drop
                # is armed immediately; otherwise it arms when the fly-in arrives.
                self._preview_center = _framing_point(row.residue)
                self._preview_settled = False
                try:
                    import numpy
                    c = self.session.main_view.center_of_rotation
                    self._preview_settled = (
                        self._preview_center is not None and c is not None and numpy.linalg.
                        norm(numpy.asarray(c, dtype=float) - self._preview_center)
                        < COFR_MATCH_DISTANCE
                    )
                except Exception:
                    pass

    def remove_preview(self, row=None):
        # Ignore stale calls from a row that no longer owns the preview.
        if row is not None and self._preview_row is not None and self._preview_row is not row:
            return
        self._stop_morph()  # a full clear also cancels any flatten-to-2D morph
        self._delete_preview_structure()
        self._preview_row = None

    def _delete_preview_structure(self):
        self._restore_replaced()
        self._preview_center = None
        self._preview_settled = False
        s = self._preview_structure
        self._preview_structure = None
        if s is not None and not s.deleted:
            s.delete()  # also removes the in-structure chain-link stub atoms/bonds

    def _drop_preview(self):
        # Fully drop the current preview: clear the owning row's grey outline and
        # armed accept button, then remove the structure. Used when the centre of
        # rotation moves off the previewed residue (the preview is no longer
        # discarded on a mere mouse-leave).
        row = self._preview_row
        if row is not None:
            try:
                row.clear_preview_outline()
            except RuntimeError:
                pass
        self.remove_preview()

    def _hide_replaced(self, residue):
        '''Hide the residue's atoms (and, with them, their bonds) while its preview
        is shown, so the preview stands in for it cleanly -- no model atoms left
        showing under/around the preview. Remembers the exact per-atom display
        state so _restore_replaced puts it back. Uses `displays` (not a hide bit)
        so it composes cleanly with spotlight masking (the separate `hides`
        bitmask). The residue's bonds to its neighbours hide with it, so
        _build_preview redraws that link as preview->neighbour pseudobonds.'''
        if residue is None or residue.deleted:
            return
        atoms = residue.atoms
        self._hidden_residue = residue
        self._saved_displays = atoms.displays  # snapshot (a fresh array)
        atoms.displays = False

    def _restore_replaced(self):
        '''Undo _hide_replaced. Re-fetches by the stored residue and checks it is
        alive + unchanged in size before restoring the saved display state; if the
        atom set changed under us (a live edit), best-effort show everything.'''
        r = self._hidden_residue
        saved = self._saved_displays
        self._hidden_residue = None
        self._saved_displays = None
        if r is None or r.deleted or saved is None:
            return
        atoms = r.atoms
        if len(atoms) == len(saved):
            atoms.displays = saved
        else:
            atoms.displays = True

    def _on_frame_drawn(self, *_):
        '''On each drawn frame -- but only when the centre of rotation actually
        moved (O(1) otherwise) -- (a) underline the listed residue now sitting at
        the centre of rotation, and (b) drop the current preview once the centre of
        rotation has moved off the residue it previews. Previews persist across
        mouse-leaves now; moving the view away is what discards them.'''
        if self._deleted or self.container.is_collapsed:
            return
        import numpy
        try:
            cofr = self.session.main_view.center_of_rotation
        except Exception:
            return
        cofr = None if cofr is None else numpy.asarray(cofr, dtype=float)
        key = b'' if cofr is None else cofr.tobytes()
        if key == self._last_cofr_key:
            return
        self._last_cofr_key = key
        # (a) underline the residue whose framing point is at the centre of rotation
        for lbl in self._name_labels:
            r = lbl.residue
            focused = False
            if cofr is not None and r is not None and not r.deleted:
                fp = _framing_point(r)
                focused = fp is not None and \
                    numpy.linalg.norm(fp - cofr) < COFR_MATCH_DISTANCE
            try:
                lbl.set_focused(focused)
            except RuntimeError:
                pass  # label destroyed under us; next populate rebuilds the list
        # (b) drop the kept preview once the centre of rotation leaves its residue,
        # but only after it has first arrived there (so the fly-in, which sweeps the
        # CofR in from afar, doesn't discard the preview it just created).
        if self._preview_center is not None and cofr is not None:
            if numpy.linalg.norm(cofr - self._preview_center) < COFR_MATCH_DISTANCE:
                self._preview_settled = True
            elif self._preview_settled:
                self._drop_preview()

    def _leaving_atom_names(self, ccd_name):
        '''CCD atom names flagged ``pdbx_leaving_atom_flag == 'Y'`` for `ccd_name`
        (empty set if unavailable). Read from ChemComp's full record, which keeps
        the flag that ccd_records/lookup drop (they return 4-tuples).'''
        try:
            from chimerax.chemcomp import record
            rec = record(self.session, ccd_name)
        except Exception:
            return set()
        if isinstance(rec, dict):
            atoms = rec.get('atoms')
            if atoms:
                return {a[0] for a in atoms if len(a) > 5 and a[5] == 'Y'}
        return set()

    @staticmethod
    def _model_carbon_color(residue):
        '''The model's carbon colour, used to paint the preview's carbons so it
        reads as part of the model. Prefers this residue's own carbons, then any
        carbon in the structure, then the element default.'''
        import numpy
        from chimerax.atomic.colors import element_colors
        c = residue.atoms[residue.atoms.element_numbers == 6]
        if len(c):
            return c.colors[0]
        sc = residue.structure.atoms
        sc = sc[sc.element_numbers == 6]
        if len(sc):
            return sc.colors[0]
        return element_colors(numpy.array([6]))[0]

    def _build_preview(self, residue, template_name):
        '''A thin-stick, element-coloured (model-carbon) copy of `template_name`'s
        ideal structure, superimposed on the residue as a child model. The atoms
        shared with the model are pinned onto the model's real coordinates and
        only the template-only atoms keep the idealised geometry -- a cheap
        half-way house between a rigid fit and a full rebuild. Returns the
        AtomicStructure, or None (any failure is non-fatal -- the outline/commit
        logic still works).'''
        if residue is None or residue.deleted:
            return None
        ccd_name = None
        try:
            import numpy
            from chimerax.atomic import Atom
            from chimerax.geometry import align_points
            from chimerax.isolde.atomic.building.place_ligand import _ccd_template_residue
            ccd_name = self._public_ccd_id(template_name)
            # No CCD reference, or a template we've already found unbuildable:
            # skip silently (don't re-fetch / re-log it on every hover).
            if ccd_name is None or self._preview_buildable.get(ccd_name) is False:
                return None
            tmpl_res = _ccd_template_residue(self.session, ccd_name)
            s = tmpl_res.structure
            res_by_name = {a.name: a for a in residue.atoms}
            pairs = [
                (ta, res_by_name[ta.name]) for ta in tmpl_res.atoms if ta.name in res_by_name
            ]
            if len(pairs) < 3:
                s.delete()
                self._preview_buildable[ccd_name] = False
                return None
            # The fit uses the residue's shared atoms; the shared backbone (N, CA,
            # C) is pinned to the model's real coordinates below, which already
            # anchors the preview onto the true chain (its N/C ARE the chain's
            # connection points). No extra neighbour anchor is needed -- and a free
            # monomer's leaving atoms sit at the template's idealised dihedral, not
            # this chain's, so using them as anchors would fight the backbone fit.
            tmpl_pts = numpy.array([ta.coord for ta, _ in pairs])
            res_pts = numpy.array([ra.coord for _, ra in pairs])
            place, _rms = align_points(tmpl_pts, res_pts)
            coords = place.transform_points(s.atoms.coords)
            rigid = coords.copy()  # rigid-fit positions, before any pinning
            index_by_atom = {a: i for i, a in enumerate(s.atoms)}
            # Half-way house between a cheap rigid fit and a full rebuild: the atoms
            # this template shares with the model are already in the model exactly
            # where the preview sits, so pin those onto the model's real coordinates.
            placed = set()
            for i, atom in enumerate(s.atoms):
                model_atom = res_by_name.get(atom.name)
                if model_atom is not None:
                    coords[i] = model_atom.coord
                    placed.add(i)
            # Re-attach the template-only atoms (no model twin) to that pinned
            # frame. Most are hydrogens whose name differs between the model and the
            # CCD, so they never name-match -- and, left at their idealised rigid-fit
            # position, they float away from a heavy neighbour that WAS pinned to a
            # different (real, often distorted) model coordinate. Carry each along by
            # the model-space shift of an already-placed neighbour, propagating
            # outward so a missing heavy atom AND the hydrogens hanging off it both
            # follow. rigid[i] - rigid[anchor] is their bond vector in the fitted
            # frame, so adding it to the anchor's placed position rebuilds the local
            # geometry off the real atom.
            pending = [i for i in range(len(s.atoms)) if i not in placed]
            progress = True
            while pending and progress:
                progress, still = False, []
                for i in pending:
                    anchor = next(
                        (
                            index_by_atom[nb]
                            for nb in s.atoms[i].neighbors if index_by_atom[nb] in placed
                        ), None
                    )
                    if anchor is None:
                        still.append(i)
                        continue
                    coords[i] = coords[anchor] + (rigid[i] - rigid[anchor])
                    placed.add(i)
                    progress = True
                pending = still
            s.atoms.coords = coords
            s.name = 'template preview'
            s.pickable = False
            s.atoms.draw_modes = Atom.STICK_STYLE
            s.atoms.radii = PREVIEW_STICK_RADIUS
            # Colour by element, with carbons taking the MODEL's carbon colour so
            # the preview reads as a natural extension of the chain rather than a
            # distinct overlay. Bonds use half-bond colouring to follow their atoms.
            from chimerax.atomic.colors import element_colors
            s.atoms.colors = element_colors(s.atoms.element_numbers)
            carbons = s.atoms[s.atoms.element_numbers == 6]
            if len(carbons):
                carbons.colors = self._model_carbon_color(residue)
            s.bonds.radii = PREVIEW_STICK_RADIUS
            s.bonds.halfbonds = True
            # The residue is hidden while its preview shows, so its own bonds to the
            # neighbouring residues vanish. Redraw that link with short stub bonds
            # INSIDE the preview structure: for each preview atom whose model twin
            # bonds across a residue boundary, add a marker atom at the real
            # neighbour's coordinate and bond the preview atom to it. Keeping both
            # ends inside the (displayed) preview means the link always draws --
            # unlike a pseudobond, whose model-side endpoint may be ribbon-hidden.
            # Best-effort: on any failure the preview simply has no link stubs.
            try:
                import numpy
                from chimerax.atomic.struct_edit import add_atom, add_bond
                prev_by_name = {a.name: a for a in s.atoms}
                link_color = self._model_carbon_color(residue)
                idx = 0
                linked_pa = []
                for ra in residue.atoms:
                    pa = prev_by_name.get(ra.name)
                    if pa is None:
                        continue
                    ext = [
                        nb for nb in ra.neighbors
                        if nb.residue is not residue and nb.element.number != 1
                    ]
                    for nb in ext:
                        stub = add_atom('lnk%d' % idx, nb.element, tmpl_res, nb.coord)
                        idx += 1
                        stub.draw_mode = Atom.STICK_STYLE
                        stub.radius = PREVIEW_STICK_RADIUS
                        stub.color = link_color
                        b = add_bond(pa, stub)
                        b.radius = PREVIEW_STICK_RADIUS
                        b.color = link_color
                    if ext:
                        linked_pa.append(pa)
                # (1a) Hide the leaving atoms at each linked position -- the peptide
                # bond replaces them, so they are absent from the polymer form. BFS
                # out from each linked atom through leaving-flagged atoms (OXT, then
                # its HXT, ...); leaving atoms at UNLINKED positions (e.g. a real
                # C-terminal OXT) are not reached and stay visible.
                leaving = self._leaving_atom_names(ccd_name)
                if leaving and linked_pa:
                    hide, frontier = set(), list(linked_pa)
                    while frontier:
                        a = frontier.pop()
                        for nb in a.neighbors:
                            if nb.name in leaving and nb not in hide:
                                hide.add(nb)
                                frontier.append(nb)
                    for a in hide:
                        a.display = False
                # (1b) Flatten linked amide nitrogens: the free monomer's N is a
                # pyramidal sp3 amine, but in the polymer it is a planar sp2 amide.
                # Put the single remaining N-H in the plane of N's two heavy
                # neighbours, opposite their bisector (~sp2).
                for pa in linked_pa:
                    if pa.element.number != 7:
                        continue
                    heavy = [
                        nb for nb in pa.neighbors if nb.element.number != 1 and nb.display
                    ]
                    hs = [nb for nb in pa.neighbors if nb.element.number == 1 and nb.display]
                    if len(heavy) < 2 or len(hs) != 1:
                        continue
                    n = numpy.asarray(pa.coord, dtype=float)
                    d1 = numpy.asarray(heavy[0].coord, dtype=float) - n
                    d2 = numpy.asarray(heavy[1].coord, dtype=float) - n
                    d1 /= (numpy.linalg.norm(d1) or 1.0)
                    d2 /= (numpy.linalg.norm(d2) or 1.0)
                    bis = d1 + d2
                    bl = numpy.linalg.norm(bis)
                    if bl >= 1e-6:
                        hs[0].coord = n - bis / bl * 1.01
            except Exception:
                pass
            residue.structure.add([s])
            self._preview_buildable[ccd_name] = True
            return s
        except Exception as e:
            # Cache the failure so it isn't retried/re-logged on every hover, and
            # warn just once (this first time) for the offending template.
            if ccd_name is not None:
                self._preview_buildable[ccd_name] = False
            self.session.logger.info(
                'New section: no preview for template "{}" ({}: {})'.format(
                    template_name, e.__class__.__name__, e
                )
            )
            return None

    # --- commit (rebuild to template) ------------------------------------
    def apply_template(self, residue, template_name):
        '''Rebuild `residue` to match the MD template (as the Unparameterised
        Residues panel's "Rebuild residue to template" does, minus the "all
        residues of this name" fan-out), then record the isolde_template_name
        override. A None template_name (grey box) just clears the override.'''
        if residue is None or residue.deleted:
            return
        if template_name is None:
            residue.isolde_template_name = None
            return
        isolde = self.isolde
        try:
            ff = isolde.forcefield_mgr[isolde.sim_params.forcefield]
            template = ff._templates[template_name]
            from .unparameterised import _get_ccd_template_and_name
            ccd, _description = _get_ccd_template_and_name(self.session, template_name)
            from chimerax.isolde.atomic.template_utils import fix_residue_to_match_md_template
            fix_residue_to_match_md_template(self.session, residue, template, cif_template=ccd)
            residue.isolde_template_name = template_name
        except Exception as e:
            self.session.logger.warning(
                'New section: could not rebuild /{}{}{} to template "{}" ({}: {})'.format(
                    residue.chain_id, residue.name, residue.number, template_name,
                    e.__class__.__name__, e
                )
            )

    def parameterise_unit(self, descriptor):
        '''Build a fresh MD template for a whole unit via the existing pipeline
        (covalent unit / metal site / free ligand), then refresh. This runs ONLY
        on the user's explicit button click -- the simulation build never
        parameterises on its own. AM1-BCC is synchronous and can take from seconds
        to minutes, so a busy cursor + status line flag the wait. On success the
        pipeline sets isolde_template_name / loads a USER_ template, so the
        residues drop out of the next detection and the next sim start matches
        them automatically.'''
        seed = descriptor['seed']
        if seed is None or seed.deleted:
            return
        kind = descriptor['kind']
        unit = descriptor['unit']
        label = self._unit_label(descriptor)
        self.session.logger.status(
            'Parameterising {} (running AM1-BCC; this may take a while)...'.format(label)
        )
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            from chimerax.isolde.openmm.amberff.covalent import (
                parameterise_metal_site,
                parameterise_covalent_unit,
                parameterise_free_ligand,
            )
            if kind == 'metal':
                parameterise_metal_site(self.session, unit)
            elif kind == 'covalent':
                parameterise_covalent_unit(self.session, unit)
            else:
                parameterise_free_ligand(self.session, seed)
        except Exception as e:
            self.session.logger.warning(
                'New section: parameterisation of {} failed ({}: {})'.format(
                    label, e.__class__.__name__, e
                )
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.session.logger.status('')
        self._refresh()

    def open_in_chemsearch(self, residue):
        '''Open `residue` in the ChimeraX-ChemSearch 2D structure editor: fetch (or
        create) its singleton panel and seed it from the residue. ChemSearch derives
        the 2D structure itself from the residue's atoms/bonds -- no conversion is
        needed here. Best-effort: a missing bundle, a deleted residue, or one
        ChemSearch cannot derive a structure from (e.g. a lone metal ion) is a
        logged warning, never an error. NOTE this replaces whatever is currently
        drawn in the shared ChemSearch panel.'''
        if residue is None or residue.deleted:
            return
        try:
            from chimerax.core.tools import get_singleton
            from chimerax.chemsearch import TOOL_NAME
            from chimerax.chemsearch.tool import ChemSearchTool
        except ImportError:
            self.session.logger.warning(
                'ChimeraX-ChemSearch is not installed; cannot open the structure '
                'editor.'
            )
            return
        try:
            tool = get_singleton(self.session, ChemSearchTool, TOOL_NAME)
            if tool is not None:
                tool.seed_from_residue(residue)
        except Exception as e:
            self.session.logger.warning(
                'New section: could not open {} in ChemSearch ({}: {})'.format(
                    residue.name, e.__class__.__name__, e
                )
            )

    # --- "flatten to 2D" morph (pencil button, default action) -----------
    def _pencil_clicked(self, residue, open_editor=False):
        '''Pencil-button action. Default (open_editor False): toggle a
        flatten-to-2D morph -- animate the residue's heavy atoms from their real
        3D positions into a 2D chemical-diagram layout, in the 3D view, so no
        second window is needed. Clicking the pencil of the residue already
        flattened restores it. Shift-click (open_editor True) opens the full
        ChemSearch 2D editor for actual editing.'''
        if residue is None or residue.deleted:
            return
        if open_editor:
            self.open_in_chemsearch(residue)
            return
        mo = self._morph
        if mo is not None and mo.get('residue') is residue:
            self._stop_morph()  # toggle the flattened depiction back off
            return
        self._start_morph(residue)

    def _2d_layout_data(self, residue):
        '''ChemSearch's 2D depiction of `residue` as plain data (see
        chemsearch.layout_2d_for_residue), or None. Never opens the ChemSearch
        tool/web view; a missing bundle or any failure is a quiet None.'''
        try:
            from chimerax.chemsearch import layout_2d_for_residue
        except Exception:
            return None
        try:
            return layout_2d_for_residue(self.session, residue)
        except Exception as e:
            self.session.logger.info(
                'New section: 2D layout for {} failed ({}: {})'.format(
                    residue.name, e.__class__.__name__, e
                )
            )
            return None

    def _start_morph(self, residue):
        '''Begin the flatten-to-2D morph for `residue`. Falls back to opening the
        full ChemSearch editor when no 2D depiction can be derived (e.g. a
        coordinated metal whose bond orders can't be perceived), so the pencil
        always does something useful.'''
        if residue is None or residue.deleted:
            return
        data = self._2d_layout_data(residue)
        if not data or not data.get('atoms'):
            self.open_in_chemsearch(residue)
            return
        # Cancel any preview/morph and restore displays BEFORE we read the
        # residue's real coordinates and hide it.
        self.remove_preview()
        try:
            mo = self._build_2d_depiction_structure(residue, data)
        except Exception as e:
            self.session.logger.info(
                'New section: 2D morph build for {} failed ({}: {})'.format(
                    residue.name, e.__class__.__name__, e
                )
            )
            mo = None
        if mo is None:
            self.open_in_chemsearch(residue)
            return
        # Hide the real residue so the flat depiction stands in for it, and select
        # it (matching the hover-preview idiom).
        self._hide_replaced(residue)
        _select_residue(residue)
        # Arm CofR-based dismissal exactly as show_preview does, so navigating away
        # drops the depiction (it faces the camera at creation, so an orbit would
        # otherwise leave it oblique). Seeded settled if we are already centred.
        self._preview_center = _framing_point(residue)
        self._preview_settled = False
        try:
            import numpy
            c = self.session.main_view.center_of_rotation
            self._preview_settled = (
                self._preview_center is not None and c is not None
                and numpy.linalg.norm(numpy.asarray(c, dtype=float) - self._preview_center)
                < COFR_MATCH_DISTANCE
            )
        except Exception:
            pass
        from Qt.QtCore import QTimer
        timer = QTimer()
        timer.timeout.connect(self._morph_tick)
        mo['timer'] = timer
        self._morph = mo
        timer.start(MORPH_TICK_MS)

    def _stop_morph(self):
        '''Tear down any active flatten-to-2D morph: stop its timer, delete the
        throwaway depiction structure, and restore the hidden residue. Idempotent;
        called by every panel exit path (remove_preview / show_preview / cleanup).'''
        mo = self._morph
        if mo is None:
            return
        self._morph = None
        t = mo.get('timer')
        if t is not None:
            try:
                t.stop()
            except Exception:
                pass
        s = mo.get('structure')
        if s is not None and not s.deleted:
            try:
                s.delete()
            except Exception:
                pass
        # Undo _hide_replaced and clear the CofR-drop arming (shared with preview).
        self._restore_replaced()
        self._preview_center = None
        self._preview_settled = False

    def _build_2d_depiction_structure(self, residue, data):
        '''Build the throwaway child structure + animation arrays for the morph,
        returning a morph-state dict or None. Coordinates are in the
        residue.structure LOCAL frame (as _build_preview), and the target 2D plane
        faces the current camera. Modelled atoms start at their real 3D position;
        unmodelled ("missing") atoms start at their 2D position with zero alpha and
        fade in as the morph completes.'''
        import numpy
        from chimerax.atomic import AtomicStructure, Atom, Atoms, Element
        from chimerax.atomic.colors import element_colors
        atoms = data.get('atoms') or []
        if not atoms:
            return None
        m = residue.structure
        # Resolve each modelled atom to its live 3D coordinate (local frame). A
        # named twin gone under a live edit is treated as unmodelled.
        twin = []
        for a in atoms:
            c = None
            nm = a.get('name')
            if a.get('modelled') and nm:
                ma = residue.find_atom(nm)
                if ma is not None and not ma.deleted:
                    c = numpy.asarray(ma.coord, dtype=float)
            twin.append(c)
        present = [c for c in twin if c is not None]
        if not present:
            return None
        centroid = numpy.mean(numpy.stack(present), axis=0)
        # 2D plane basis = camera right/up, expressed in the model's local frame.
        cam = self.session.main_view.camera
        cmat = numpy.asarray(cam.position.matrix, dtype=float)[:, :3]
        right, up = cmat[:, 0], cmat[:, 1]
        try:
            R = numpy.asarray(m.scene_position.inverse().matrix, dtype=float)[:, :3]
            right, up = R.dot(right), R.dot(up)
        except Exception:
            pass
        right = right / (numpy.linalg.norm(right) or 1.0)
        up = up / (numpy.linalg.norm(up) or 1.0)
        xy = numpy.array([(a['x'], a['y']) for a in atoms], dtype=float)
        lc = xy.mean(axis=0)
        target = (
            centroid[None, :] + DEPICTION_SCALE * (
                (xy[:, 0] - lc[0])[:, None] * right[None, :] +
                (xy[:, 1] - lc[1])[:, None] * up[None, :]
            )
        )
        start = numpy.array(
            [twin[i] if twin[i] is not None else target[i] for i in range(len(atoms))],
            dtype=float
        )
        missing = numpy.array([twin[i] is None for i in range(len(atoms))], dtype=bool)
        # Throwaway depiction structure (heavy atoms only), initially overlaying the
        # real atoms; missing atoms start invisible. Keep an atom collection in the
        # SAME order as `atoms`/`start`/... so per-frame coord/colour assignment
        # lines up (do not rely on Structure.atoms ordering).
        s = AtomicStructure(self.session, name='2D depiction', auto_style=False)
        dres = s.new_residue(residue.name, 'A', 1)
        sa = []
        for i, a in enumerate(atoms):
            na = s.new_atom('{}{}'.format(a['element'], i), Element.get_element(a['element']))
            na.coord = start[i]
            dres.add_atom(na)
            sa.append(na)
        for b in data.get('bonds') or []:
            i, j = b.get('i'), b.get('j')
            if i is not None and j is not None and i != j \
                    and 0 <= i < len(sa) and 0 <= j < len(sa):
                try:
                    s.new_bond(sa[i], sa[j])
                except Exception:
                    pass
        s.pickable = False
        coll = Atoms(sa)
        coll.draw_modes = Atom.STICK_STYLE
        coll.radii = PREVIEW_STICK_RADIUS
        base_colors = element_colors(coll.element_numbers)
        carbons = coll.element_numbers == 6
        if carbons.any():
            base_colors[carbons] = self._model_carbon_color(residue)
        base_colors[:, 3] = 255
        init = base_colors.copy()
        init[missing, 3] = 0
        coll.colors = init
        s.bonds.radii = PREVIEW_STICK_RADIUS
        s.bonds.halfbonds = True
        m.add([s])
        return {
            'structure': s,
            'residue': residue,
            'coll': coll,
            'start': start,
            'target': target,
            'base_colors': base_colors,
            'missing': missing,
            't': 0.0,
        }

    def _morph_tick(self):
        '''One animation step: ease the depiction from 3D (t=0) to 2D (t=1) and
        fade the missing atoms in. Stops the timer at t=1 (the flat depiction then
        persists until dismissed). Bails safely if the structure or residue was
        deleted under us (live editing).'''
        mo = self._morph
        if mo is None:
            return
        s = mo.get('structure')
        residue = mo.get('residue')
        if s is None or s.deleted or residue is None or residue.deleted:
            self._stop_morph()
            return
        mo['t'] = min(1.0, mo['t'] + MORPH_STEP)
        e = _ease_in_out_sine(mo['t'])
        try:
            coll = mo['coll']
            coll.coords = mo['start'] + e * (mo['target'] - mo['start'])
            colors = mo['base_colors'].copy()
            colors[mo['missing'], 3] = int(round(e * 255))
            coll.colors = colors
        except Exception:
            self._stop_morph()
            return
        if mo['t'] >= 1.0:
            t = mo.get('timer')
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass

    # --- data (FMCS-scored candidate templates) --------------------------
    def _detect(self):
        '''Bulk step: the selected model's unparameterised residues, CLUSTERED into
        the units the parameterisation pipeline builds, WITHOUT the per-candidate
        FMCS. Returns a list of entries, each either:

          * ``('unit', descriptor)`` -- a covalent unit or metal site (rendered as
            one "Parameterise unit" row); or
          * ``('residue', residue, kind, payload, descriptor)`` -- a single free
            residue (the existing candidate-box flow), where kind is 'unmatched'
            (payload = the OpenMM residue, for find_possible_templates) or
            'ambiguous' (payload = the competing template_info), and descriptor is
            its 'free' grouping descriptor (None only if grouping itself failed).

        [] if none / no model. Detection + grouping only -- no side effects.'''
        isolde = self.isolde
        m = isolde.selected_model
        if m is None:
            return []
        try:
            from chimerax.atomic import Residues
            residues = Residues(
                sorted(m.residues, key=lambda r: (r.chain_id, r.number, r.insertion_code))
            )
            ffmgr = isolde.forcefield_mgr
            ff = ffmgr[isolde.sim_params.forcefield]
            ligand_db = ffmgr.ligand_db(isolde.sim_params.forcefield)
            from chimerax.isolde.openmm.openmm_interface import (
                find_residue_templates, create_openmm_topology
            )
            template_dict = find_residue_templates(
                residues, ff, ligand_db=ligand_db, logger=self.session.logger
            )
            top, residue_templates = create_openmm_topology(residues.atoms, template_dict)
            _, ambiguous, unmatched = ff.assignTemplates(
                top, ignoreExternalBonds=True, explicit_templates=residue_templates
            )
        except Exception as e:
            self.session.logger.info(
                'New section: could not determine unparameterised residues '
                '({}: {})'.format(e.__class__.__name__, e)
            )
            return []
        # Map the OpenMM offenders back to ChimeraX residues, keeping each one's
        # per-residue payload for the free-ligand FMCS candidate lookup.
        payload_by_residue = {}
        for r in unmatched:
            payload_by_residue[residues[r.index]] = ('unmatched', r)
        for r, tinfo in ambiguous.items():
            payload_by_residue[residues[r.index]] = ('ambiguous', tinfo)
        offenders = list(payload_by_residue.keys())
        if not offenders:
            return []
        # Context-aware grouping: cluster the isolated failures into the units the
        # pipeline actually builds (metal site / covalent unit / free ligand), so a
        # covalent ligand + its partner residue is ONE entry rather than several.
        # Cheap (graph/geometry only); no model mutation.
        try:
            from chimerax.isolde.openmm.amberff.covalent import (group_for_parameterisation)
            groups = group_for_parameterisation(self.session, offenders, forcefield=ff)
        except Exception as e:
            self.session.logger.info(
                'New section: could not group unparameterised residues; falling '
                'back to per-residue ({}: {})'.format(e.__class__.__name__, e)
            )
            return [('residue', r, k, p, None) for r, (k, p) in payload_by_residue.items()]
        entries = []
        for g in groups:
            if g['kind'] in ('metal', 'covalent'):
                entries.append(('unit', g))
            else:  # free -> keep the per-residue candidate / rebuild flow
                r = g['seed']
                kind, payload = payload_by_residue.get(r, ('unmatched', None))
                entries.append(('residue', r, kind, payload, g))
        return entries

    def _candidates_for(self, kind, payload, residue):
        '''(name_cands, comp_cands) for one residue -- the expensive per-candidate
        FMCS step, run lazily from _process_next so rows appear progressively.'''
        res_mol = self._residue_mol(residue)
        if kind == 'unmatched':
            try:
                ff = self.isolde.forcefield_mgr[self.isolde.sim_params.forcefield]
                by_name, by_comp = ff.find_possible_templates(payload)
            except Exception:
                by_name, by_comp = [], []
            return (
                self._candidates([tn for tn, _ in by_name], 'Name Match', res_mol),
                self._candidates([tn for tn, _ in by_comp], 'Topology Match', res_mol),
            )
        names = [ti[0].name for ti in payload]
        return ([], self._candidates(names, 'Topology Match', res_mol))

    def _candidates(self, template_names, kind, res_mol):
        '''Candidate descriptors for a group, ordered by descending FMCS overlap
        (best match leftmost); unscorable candidates (None) sort last.'''
        cands = [self._candidate(tn, kind, res_mol) for tn in template_names]
        cands.sort(
            key=lambda c: c['fraction'] if c['fraction'] is not None else -1.0, reverse=True
        )
        return cands

    def _candidate(self, tname, kind, res_mol):
        frac = self._fmcs_fraction(res_mol, tname)
        from .unparameterised import _get_ccd_template_and_name
        ccd, description = _get_ccd_template_and_name(self.session, tname)
        ccd_name = ccd.name if ccd is not None else 'Not found'
        return {
            'template_name': tname,
            'fraction': frac,
            'fill': _fraction_to_viridis_css(frac),
            'tooltip': _match_tooltip(kind, tname, frac, ccd_name, description),
        }

    def _residue_mol(self, residue):
        '''Context-aware heavy-atom RDKit Mol for `residue`, built once per
        residue. Uses super_residue_to_rdkit([residue]), which replaces every
        bond leaving the residue with a cap (ACE/NME on a peptide backbone cut,
        methyl otherwise) at the real neighbour's position -- so a residue that
        is a small part of a larger molecule keeps correct valence at its seam
        atoms instead of being mis-perceived in isolation. Cap atoms carry a
        'CAP' property so the FMCS fraction can exclude them. Falls back to the
        isolated residue_to_rdkit build, then None.'''
        try:
            from chimerax.isolde.atomic import rdkit_bridge as rb
            from rdkit import Chem
            mol, _cxmap, _info = rb.super_residue_to_rdkit([residue])
            if mol is not None:
                # FMCS compares heavy-atom topology; the CCD template mols are
                # heavy-only, so drop the (modelled + added) hydrogens to match.
                try:
                    return Chem.RemoveHs(mol)
                except Exception:
                    return Chem.RemoveHs(mol, sanitize=False)
        except Exception:
            pass
        try:
            from chimerax.isolde.atomic import rdkit_bridge as rb
            mol, _atom_map = rb.residue_to_rdkit(residue)
            return mol
        except Exception:
            return None

    @staticmethod
    def _is_real_atom(atom):
        # Real residue atoms (not caps added by super_residue_to_rdkit).
        return not atom.HasProp('CAP')

    def _fmcs_fraction(self, res_mol, tname):
        '''RDKit FMCS overlap between the residue and template `tname`'s CCD
        reference, in [0, 1]. Counts only the residue's REAL atoms (cap atoms
        added for the polymer context are excluded from both the match and the
        denominator), so caps improve valence without inflating the size:
        matched_real / max(real residue atoms, template atoms). None if
        uncomputable.'''
        if res_mol is None:
            return None
        try:
            ccd_name = self._public_ccd_id(tname)
            if ccd_name is None:
                return None
            tmpl_mol = self._template_mol(ccd_name)
            if tmpl_mol is None:
                return None
            from chimerax.isolde.atomic import rdkit_bridge as rb
            corr = rb.fmcs_index_correspondence(res_mol, tmpl_mol, timeout=FMCS_TIMEOUT_S)
            n_res = sum(1 for a in res_mol.GetAtoms() if self._is_real_atom(a))
            matched = sum(
                1 for ri, _ti in corr if self._is_real_atom(res_mol.GetAtomWithIdx(ri))
            )
            n = max(n_res, tmpl_mol.GetNumAtoms())
            return matched / n if n else None
        except Exception:
            return None

    def _public_ccd_id(self, template_name):
        '''Public CCD id to use as the reference for a candidate MD template.

        template_name_to_ccd_name maps templates to ISOLDE-internal variant ids
        (e.g. CYX -> "CYS_LL_DHG", ALA -> "ALA_LL") that only exist in ISOLDE's
        private collection. For our purpose -- topological FMCS + a rough
        superimposed preview -- the base chemistry is enough, and the parent is
        the leading token before "_" ("CYS_LL_DHG" -> "CYS"). Public CCD ids have
        no underscore, so this is a no-op for a real CCD id / ligand code. When
        the map has no entry, the template name itself is usually the CCD id
        (ligands), so fall back to it.'''
        from chimerax.isolde.openmm.amberff.template_utils import (template_name_to_ccd_name)
        ccd_name, _extra = template_name_to_ccd_name(template_name)
        if ccd_name is None:
            ccd_name = template_name
        if not ccd_name:
            return None
        return ccd_name.split('_')[0]

    def _template_mol(self, ccd_name):
        '''Memoised RDKit mol for a CCD component (None if unavailable). Keyed by
        ccd_name so each component is fetched at most once, sparing the repeated
        CCD network-fetches (and their log spam) across candidates and hovers.'''
        if ccd_name in self._tmpl_mol_cache:
            return self._tmpl_mol_cache[ccd_name]
        mol = None
        try:
            from chimerax.isolde.atomic import rdkit_bridge as rb
            mol, _status = rb.template_to_rdkit(self.session, ccd_name)
        except Exception:
            mol = None
        self._tmpl_mol_cache[ccd_name] = mol
        return mol

    def selected_model_changed_cb(self, *_):
        # Switching models drops any preview and clears the list; it repopulates
        # on the next detection or when the section is (re)expanded.
        self._clear_rows()
