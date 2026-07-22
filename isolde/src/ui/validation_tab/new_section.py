# @Author: Tristan Croll
# @Date:   21-Jul-2026
# @Email:  tcroll@altoslabs.com
# @Last modified by:   tcroll
# @Last modified time: 21-Jul-2026
# @License: Free for non-commercial use (see license.pdf)
# @Copyright: 2026 Tristan Croll
'''
Scratch/prototype validation section, "next rotamer"-style. One entry per
unparameterised residue in the selected model (labelled "<name>, chain <X>"),
each a row of: an arrow "cycle" button, a grey "no imposed template" box, then
one box per candidate MD template grouped "by residue name" then "by topology
similarity", ordered best-first. Template boxes are coloured by RDKit FMCS
overlap with the residue (green = identical, purple-blue = poor); their tooltip
carries the "possible templates" details with that FMCS overlap.

Two outlines: a RED box marks the committed choice (applied to the model); a GREY
box marks a transient *preview*. A preview is a thin-stick copy of the template's
ideal structure, aligned onto the residue and superimposed -- low-commitment, it
does not touch the model. Interaction:
  * Hovering a box (>~2 frames, no button) previews it (grey) and flies the
    camera to the residue; leaving the box drops the preview. Red is unchanged.
  * Clicking a box moves the red box and rebuilds the residue to that template
    (the grey box clears the override).
  * Clicking the arrow cycles the grey preview to the next box (wrapping);
    moving the mouse off the arrow commits the previewed box to the model.

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
)
from Qt.QtCore import Qt, QTimer

from ..collapse_button import CollapsibleArea
from ..ui_base import UI_Panel_Base, DefaultVLayout, DefaultHLayout

# matplotlib is already a hard ISOLDE dependency (see the Ramachandran plot), so
# sampling viridis for the score colour ramp adds no new requirement.
from matplotlib import colormaps

_VIRIDIS = colormaps['viridis']

BOX_SIZE = 20  # px, each box is a fixed square
GROUP_GAP = 14  # px, extra space inserted between groups
# Show roughly this many rows before the vertical scrollbar kicks in.
VISIBLE_ROWS = 8

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
# Hover dwell before a box previews / flies. Approximated in ms (>~2 frames)
# because a Qt-panel hover doesn't reliably tick the GL 'new frame' trigger.
HOVER_PREVIEW_DELAY_MS = 50
# A residue whose centroid sits within this off-axis fraction (tan of the angle
# from the camera's view axis) counts as "already centred" -- the hover-fly then
# skips, leaving any manual reorientation untouched. ~0.12 ~= 7 degrees.
CAMERA_CENTERED_FRACTION = 0.12
# FMCS is NP-hard; cap each residue-vs-template comparison so a large/symmetric
# residue can't stall the panel. RDKit returns its best match so far on timeout.
FMCS_TIMEOUT_S = 2
# Preview thin-stick appearance.
PREVIEW_STICK_RADIUS = 0.05
PREVIEW_COLOR = (255, 215, 0, 255)  # gold, to read clearly over the model


def _ease_in_out_sine(t):
    '''Ease-in-out on a fraction t in [0, 1]: zero derivative (hence zero
    linear and angular camera velocity) at both ends, peak rate at the middle.'''
    return (1.0 - cos(pi * t)) / 2.0


def _residue_is_centred(residue):
    '''True if the residue is already near the centre of the view (its centroid
    within CAMERA_CENTERED_FRACTION of the view axis, in front of the camera).
    Orientation-independent, so a user who reoriented while keeping the residue
    framed is judged already-positioned and left undisturbed.'''
    import numpy
    atoms = residue.atoms
    if not len(atoms):
        return False
    centroid = atoms.scene_coords.mean(axis=0)
    cam = residue.structure.session.main_view.camera
    to_res = centroid - cam.position.origin()
    view_dir = cam.view_direction()
    depth = float(numpy.dot(to_res, view_dir))  # along the axis; >0 is in front
    if depth <= 0:
        return False
    perp = numpy.linalg.norm(to_res - depth * view_dir)  # distance off the axis
    return perp / depth < CAMERA_CENTERED_FRACTION


def _fly_to_residue(residue):
    '''Smoothly move the camera to view `residue` in its standard orientation
    (ISOLDE's ResidueStepper) with an ease-in-out curve. Skips when the residue
    is already centred, preserving a manual reorientation.'''
    if residue is None or residue.deleted or _residue_is_centred(residue):
        return
    from ... import navigate
    stepper = navigate.get_stepper(residue.structure)
    stepper.step_to(
        residue,
        easing=_ease_in_out_sine,
        frames=TRANSITION_FRAMES,
        max_interpolate_distance=float('inf')
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
    '''A fixed-size coloured square for one candidate template (or the grey "no
    imposed template" box). Reports clicks/hovers to its BoxRow. Carries two
    independent outline states: committed (red) and previewed (grey); red wins.'''

    _COMMITTED_BORDER = '3px solid #e53935'  # red: the applied choice
    _PREVIEW_BORDER = '3px solid #9e9e9e'  # grey: the transient preview
    _PLAIN_BORDER = '1px solid rgba(0, 0, 0, 90)'

    def __init__(self, row, index, fill_css, tooltip, template_name, parent=None):
        super().__init__(parent)
        self.row = row
        self.index = index
        self.template_name = template_name  # None for the grey "no template" box
        self._fill_css = fill_css
        self.setFixedSize(BOX_SIZE, BOX_SIZE)
        self._committed = False
        self._previewed = False
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
        # Hover (no button) previews after a short dwell; see _maybe_preview.
        if QApplication.mouseButtons() == Qt.MouseButton.NoButton:
            QTimer.singleShot(HOVER_PREVIEW_DELAY_MS, self._maybe_preview)
        super().enterEvent(event)

    def leaveEvent(self, event):
        try:
            self.row.hover_leave(self.index)
        except RuntimeError:
            pass
        super().leaveEvent(event)

    def _maybe_preview(self):
        # Deferred hover: only preview if still hovered with no button held.
        # Guarded against the box being deleted before the timer fires.
        try:
            still_hovered = self.underMouse()
        except RuntimeError:
            return
        if still_hovered and QApplication.mouseButtons() == Qt.MouseButton.NoButton:
            self.row.hover_preview(self.index)


class ArrowButton(QToolButton):
    '''The per-row "cycle" button. Clicking advances the grey preview to the next
    box (wrapping); moving the mouse off the button commits the previewed box.'''

    def __init__(self, row, parent=None):
        super().__init__(parent)
        self.row = row
        self.setArrowType(Qt.RightArrow)
        self.setAutoRaise(True)
        self.setFixedSize(BOX_SIZE, BOX_SIZE)
        self.clicked.connect(self._clicked)

    def _clicked(self, *_):
        self.row.arrow_clicked()

    def leaveEvent(self, event):
        try:
            self.row.arrow_left()
        except RuntimeError:
            pass
        super().leaveEvent(event)


class BoxRow(QWidget):
    '''One residue's row: arrow, grey box, then viridis template boxes. Owns the
    committed (red) and preview (grey) indices and drives the dialog's preview /
    commit for this residue.'''

    def __init__(self, name_cands, comp_cands, residue=None, dialog=None, parent=None):
        super().__init__(parent)
        # Held across time -- always check `.deleted` before use.
        self.residue = residue
        self._dialog = dialog
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        hl = DefaultHLayout()
        hl.setSpacing(2)
        # Cycle arrow, between the residue label and the first (grey) box.
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

    # --- outline state ---------------------------------------------------
    def _refresh_outlines(self):
        for i, b in enumerate(self.boxes):
            b.set_committed(i == self._committed_index)
            b.set_previewed(i == self._preview_index)

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
        self._arrow_armed = False
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
        # untouched. A box hover ends any arrow-cycle.
        self._arrow_armed = False
        self._set_preview(index)
        _fly_to_residue(self.residue)

    def hover_leave(self, index):
        # Leaving a hovered box drops the transient preview -- unless an
        # arrow-cycle owns the current preview (which commits on arrow-leave).
        if not self._arrow_armed and self._preview_index == index:
            self._set_preview(None)

    # --- box click -------------------------------------------------------
    def click(self, index):
        # Commit: red moves here, model is rebuilt; any preview is dropped.
        self._arrow_armed = False
        self._set_preview(None)
        self._commit(index)

    # --- arrow -----------------------------------------------------------
    def arrow_clicked(self):
        # Advance the grey preview to the next box, wrapping. Red is untouched.
        start = self._committed_index if self._preview_index is None \
            else self._preview_index
        self._arrow_armed = True
        self._set_preview((start + 1) % len(self.boxes))

    def arrow_left(self):
        # Moving off the arrow after cycling commits the previewed box.
        if self._arrow_armed and self._preview_index is not None:
            index = self._preview_index
            self._arrow_armed = False
            self._set_preview(None)
            self._commit(index)


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
        self._preview_row = None
        self._preview_structure = None
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

    def _refresh(self, *_):
        if self.container.is_collapsed:
            self._dirty = True
            return
        self._dirty = False
        self.remove_preview()
        self._rebuild(self._residue_template_options())

    def _rebuild(self, options):
        self._clear_rows()
        grid = self._grid
        for i, (residue, name_cands, comp_cands) in enumerate(options):
            text = '{}, chain {}'.format(residue.name, residue.chain_id)
            grid.addWidget(QLabel(text), i, 0, Qt.AlignmentFlag.AlignVCenter)
            row = BoxRow(name_cands, comp_cands, residue=residue, dialog=self)
            grid.addWidget(row, i, 1, Qt.AlignmentFlag.AlignVCenter)
            self.rows.append(row)
        self._size_scroll(len(self.rows))

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

    # --- preview management (one superimposed preview at a time) ----------
    def show_preview(self, row, template_name):
        # A different row taking over clears the previous row's grey outline.
        if self._preview_row is not None and self._preview_row is not row:
            self._preview_row.clear_preview_outline()
        self._delete_preview_structure()
        self._preview_row = row
        if template_name is not None:
            self._preview_structure = self._build_preview(row.residue, template_name)

    def remove_preview(self, row=None):
        # Ignore stale calls from a row that no longer owns the preview.
        if row is not None and self._preview_row is not None and self._preview_row is not row:
            return
        self._delete_preview_structure()
        self._preview_row = None

    def _delete_preview_structure(self):
        s = self._preview_structure
        self._preview_structure = None
        if s is not None and not s.deleted:
            s.delete()

    def _build_preview(self, residue, template_name):
        '''A thin-stick copy of `template_name`'s ideal structure, aligned onto
        the residue by shared atom names and added as a child so it superimposes
        and moves with the model. Returns the AtomicStructure, or None (any
        failure is non-fatal -- the outline/commit logic still works).'''
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
            tmpl_pts = numpy.array([ta.coord for ta, _ in pairs])
            res_pts = numpy.array([ra.coord for _, ra in pairs])
            place, _rms = align_points(tmpl_pts, res_pts)
            s.atoms.coords = place.transform_points(s.atoms.coords)
            s.name = 'template preview'
            s.pickable = False
            s.atoms.draw_modes = Atom.STICK_STYLE
            s.atoms.radii = PREVIEW_STICK_RADIUS
            s.atoms.colors = PREVIEW_COLOR
            s.bonds.radii = PREVIEW_STICK_RADIUS
            s.bonds.colors = PREVIEW_COLOR
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

    # --- data (FMCS-scored candidate templates) --------------------------
    def _residue_template_options(self):
        '''For each unparameterised residue in the selected model, return
        (residue, name_cands, comp_cands). find_possible_templates supplies the
        candidate template *names* (its match score is ignored); each is scored,
        coloured and ordered by RDKit FMCS overlap with the residue. [] if none /
        no model. No side effects (in particular no add-hydrogens prompt).'''
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
            out = []
            for r in unmatched:
                cx_res = residues[r.index]
                res_mol = self._residue_mol(cx_res)
                by_name, by_comp = ff.find_possible_templates(r)
                out.append(
                    (
                        cx_res,
                        self._candidates([tn for tn, _ in by_name], 'Name Match', res_mol),
                        self._candidates([tn for tn, _ in by_comp], 'Topology Match', res_mol),
                    )
                )
            for r, template_info in ambiguous.items():
                cx_res = residues[r.index]
                res_mol = self._residue_mol(cx_res)
                names = [ti[0].name for ti in template_info]
                out.append((cx_res, [], self._candidates(names, 'Topology Match', res_mol)))
        except Exception as e:
            self.session.logger.info(
                'New section: could not determine unparameterised residues '
                '({}: {})'.format(e.__class__.__name__, e)
            )
            return []
        return out

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
