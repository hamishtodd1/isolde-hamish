# @Author: Tristan Croll
# @Date:   27-Jul-2026
# @Email:  tcroll@altoslabs.com
# @Last modified by:   tcroll
# @Last modified time: 27-Jul-2026
# @License: Free for non-commercial use (see license.pdf)
# @Copyright: 2026 Tristan Croll
'''
Headless characterisation test for ``group_for_parameterisation``
(``chimerax.isolde.openmm.amberff.covalent``) -- the context-aware clustering
that turns the residues which fail *isolated* template matching into the units
the parameterisation pipeline actually builds: metal sites, covalent units and
free ligands.

It is the classification half of ``parameterise_cmd``, factored out so the
validation panel and the command share one routing implementation. It does only
graph/geometry work (``detect_metal_site`` / ``detect_covalent_unit``) -- no
ANTECHAMBER/AM1-BCC and no model mutation -- so it runs headless. Run inside
ChimeraX:

    run_chimerax.bat --nogui --exit --script src/tests/test_param_grouping.py

Prints PASS/FAIL and exits non-zero on failure.
'''
import os

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '1pmx_1.pdb')


def _fail(msg):
    print('FAIL: %s' % msg)
    raise SystemExit(1)


def run(session):
    from chimerax.core.commands import run as run_cmd
    try:
        from rdkit import Chem  # noqa: F401  (detect_* build RDKit mols downstream)
    except ImportError:
        print('SKIP: RDKit not available')
        return
    import numpy
    from chimerax.isolde.openmm.amberff import covalent as cov

    m = run_cmd(session, 'open %s' % FIXTURE)[0]
    try:
        existing = [r for r in m.chains[0].residues if r is not None]
        cb_hosts = [r for r in existing if 'CB' in set(r.atoms.names)]
        if len(cb_hosts) < 5:
            _fail('fixture has too few CB-bearing residues for the test')
        # Use only "plain" hosts -- residues the detector already rejects (no
        # pre-existing non-standard linkage, so NOT a native disulfide cysteine)
        # that also have a backbone neighbour -- so each synthetic bond below
        # yields a clean, isolated unit rather than merging with real chemistry.
        from chimerax.core.errors import UserError

        def _plain(r):
            try:
                cov.detect_covalent_unit(r)
                return False
            except UserError:
                return True

        plain = [r for r in cb_hosts if len(r.neighbors) >= 1 and _plain(r)]
        if len(plain) < 5:
            _fail('fixture has too few plain CB-bearing residues for the test')
        k = len(plain)
        h_lig, h_x1, h_x2, h_mc = plain[0], plain[k // 4], plain[k // 2], plain[3 * k // 4]
        if len({h_lig, h_x1, h_x2, h_mc}) != 4:
            _fail('could not pick 4 distinct host residues')
        h_plain = next((r for r in plain if r not in {h_lig, h_x1, h_x2, h_mc}), None)
        if h_plain is None:
            _fail('no spare plain residue for the had_neighbours case')

        def _bond_new_ligand(resnum, anchor_atom, name='LIG'):
            lg = m.new_residue(name, 'Z', resnum)
            c = m.new_atom('C1', 'C')
            c.coord = numpy.array(anchor_atom.coord) + numpy.array([1.5, 0.0, 0.0])
            lg.add_atom(c)
            m.new_bond(anchor_atom, c)
            return lg

        # (a) covalent ligand bonded to a sidechain; (b) standard-standard
        # crosslink; (c) main-chain modification.
        lig = _bond_new_ligand(901, h_lig.find_atom('CB'), 'LIG')
        m.new_bond(h_x1.find_atom('CB'), h_x2.find_atom('CB'))
        ligm = _bond_new_ligand(904, h_mc.find_atom('N'), 'LGM')

        # (d) free ligand, unbonded, placed well away from everything.
        free = m.new_residue('FRE', 'Z', 905)
        fa = m.new_atom('C1', 'C')
        fa.coord = numpy.array([300.0, 300.0, 300.0])
        free.add_atom(fa)

        # (e) metal site: a Zn with two N donors ~2.1 A away, isolated in space.
        znr = m.new_residue('ZN', 'Z', 906)
        zn = m.new_atom('ZN', 'Zn')
        zn.coord = numpy.array([500.0, 500.0, 500.0])
        znr.add_atom(zn)
        lmet = m.new_residue('LMT', 'Z', 907)
        for nm, off in (('N1', [2.1, 0, 0]), ('C1', [3.0, 1.0, 0]), ('N2', [0, 2.1, 0])):
            a = m.new_atom(nm, nm[0])
            a.coord = numpy.array([500.0, 500.0, 500.0]) + numpy.array(off)
            lmet.add_atom(a)
        m.new_bond(lmet.find_atom('N1'), lmet.find_atom('C1'))
        m.new_bond(lmet.find_atom('C1'), lmet.find_atom('N2'))

        # --- Part A: classification of a mixed offender set ----------------
        offenders = [lig, h_x1, h_x2, ligm, free, znr, lmet]
        groups = cov.group_for_parameterisation(session, offenders)
        by_seed = {g['seed']: g for g in groups}
        # One descriptor per unit: h_x2 is absorbed by h_x1, lmet by znr.
        if len(groups) != 5:
            _fail('expected 5 unit descriptors, got %d (%r)'
                  % (len(groups), [(g['kind'], g['seed'].name) for g in groups]))
        gl = by_seed.get(lig)
        if gl is None or gl['kind'] != 'covalent' or set(gl['residues']) != {lig, h_lig}:
            _fail('covalent ligand attachment mis-grouped: %r' % (gl,))
        gx = by_seed.get(h_x1)
        if gx is None or gx['kind'] != 'covalent' or set(gx['residues']) != {h_x1, h_x2}:
            _fail('standard-standard crosslink mis-grouped: %r' % (gx,))
        if h_x2 in by_seed:
            _fail('crosslink partner should be absorbed, not its own descriptor')
        gm = by_seed.get(ligm)
        if gm is None or gm['kind'] != 'covalent' or set(gm['residues']) != {ligm, h_mc}:
            _fail('main-chain modification mis-grouped: %r' % (gm,))
        gf = by_seed.get(free)
        if gf is None or gf['kind'] != 'free' or set(gf['residues']) != {free}:
            _fail('free ligand mis-grouped: %r' % (gf,))
        gz = by_seed.get(znr)
        if gz is None or gz['kind'] != 'metal' or set(gz['residues']) != {znr, lmet}:
            _fail('metal site mis-grouped: %r' % (gz,))
        if lmet in by_seed:
            _fail('metal donor ligand should be absorbed, not its own descriptor')
        if gz.get('unsupported') is not None:
            _fail('with no forcefield, unsupported must be None: %r' % gz['unsupported'])
        print('PASS: group_for_parameterisation classifies metal / covalent / free units')

        # --- Part B: partition -- every input in exactly one descriptor ----
        for r in offenders:
            hits = [g for g in groups if r in g['residues']]
            if len(hits) != 1:
                _fail('%s%d appears in %d descriptors (want 1)'
                      % (r.name, r.number, len(hits)))
        print('PASS: every input residue lands in exactly one unit')

        # --- Part C: too_big (metal/covalent only; free never flagged) -----
        big = cov.group_for_parameterisation(session, [lig], max_heavy_atoms=1)
        if not big or not big[0]['too_big']:
            _fail('a covalent unit above max_heavy_atoms should be flagged too_big')
        freebig = cov.group_for_parameterisation(session, [free], max_heavy_atoms=0)
        if not freebig or freebig[0]['too_big']:
            _fail('a free ligand must never be flagged too_big')
        print('PASS: too_big flags oversized metal/covalent units (never free ligands)')

        # --- Part D: metal-involved-but-unresolvable stays a metal ---------
        # A lone metal with no donors must NOT fall through to covalent/free
        # (GAFF2 has no metal type); it stays a metal descriptor carrying the error.
        znr2 = m.new_residue('ZN', 'Z', 908)
        zn2 = m.new_atom('ZN', 'Zn')
        zn2.coord = numpy.array([900.0, 900.0, 900.0])
        znr2.add_atom(zn2)
        gsolo = cov.group_for_parameterisation(session, [znr2])
        if len(gsolo) != 1 or gsolo[0]['kind'] != 'metal':
            _fail('a lone metal should stay in the metal category')
        if gsolo[0]['unit'] is not None or not gsolo[0]['error']:
            _fail('an unresolved metal site should carry unit=None + an error')
        print('PASS: metal with no donors -> metal descriptor (unit=None + error)')

        # --- Part E: plain bonded residue -> free with had_neighbours ------
        gp = cov.group_for_parameterisation(session, [h_plain])
        if len(gp) != 1 or gp[0]['kind'] != 'free' or not gp[0]['had_neighbours']:
            _fail('a plain bonded residue should be free with had_neighbours=True')
        print('PASS: plain bonded residue -> free (had_neighbours), not covalent')

        # --- Part F: unsupported-metal pre-check (needs the ion-aware ff) --
        # With the forcefield carrying ISOLDE's ion LJ set, a supported metal
        # (Zn2+) is buildable while an unsupported one (Mo) is flagged up front,
        # so callers can refuse it before the slow AM1-BCC run.
        from chimerax.isolde.openmm.amberff import amber_convert as _ac
        from openmm.app import ForceField as _OMMFF
        _ad = os.path.dirname(_ac.__file__)
        ff = _OMMFF(os.path.join(_ad, 'amberff14SB.xml'),
                    os.path.join(_ad, 'gaff2.xml'),
                    os.path.join(_ad, 'tip3p_standard.xml'),
                    os.path.join(_ad, 'tip3p_HFE_multivalent.xml'))
        gz_ff = cov.group_for_parameterisation(session, [znr], forcefield=ff)
        if len(gz_ff) != 1 or gz_ff[0]['unsupported']:
            _fail('supported metal (Zn) wrongly flagged unsupported: %r' % (gz_ff,))
        mor = m.new_residue('MO', 'Z', 909)
        mo = m.new_atom('MO', 'Mo')
        mo.coord = numpy.array([700.0, 700.0, 700.0])
        mor.add_atom(mo)
        lmo = m.new_residue('LMO', 'Z', 910)
        for nm, off in (('N1', [2.1, 0, 0]), ('C1', [3.0, 1.0, 0]), ('N2', [0, 2.1, 0])):
            a = m.new_atom(nm, nm[0])
            a.coord = numpy.array([700.0, 700.0, 700.0]) + numpy.array(off)
            lmo.add_atom(a)
        m.new_bond(lmo.find_atom('N1'), lmo.find_atom('C1'))
        m.new_bond(lmo.find_atom('C1'), lmo.find_atom('N2'))
        gmo = cov.group_for_parameterisation(session, [mor], forcefield=ff)
        if len(gmo) != 1 or gmo[0]['kind'] != 'metal':
            _fail('Mo site should group as a metal unit: %r' % (gmo,))
        if not gmo[0]['unsupported'] or 'Mo' not in gmo[0]['unsupported']:
            _fail('Mo (no bundled LJ) should be flagged unsupported: %r' % (gmo[0],))
        print('PASS: unsupported metal (Mo) flagged up front; supported (Zn) is not')

        # --- Part G: solvent is never a parameterisation unit -------------
        # Water is handled by ISOLDE's water model, never this pipeline; an
        # (H-less) crystal water must not be offered/attempted here.
        wat = m.new_residue('HOH', 'Z', 911)
        wo = m.new_atom('O', 'O')
        wo.coord = numpy.array([320.0, 320.0, 320.0])
        wat.add_atom(wo)
        if cov.group_for_parameterisation(session, [wat]):
            _fail('a water (HOH) must never become a parameterisation unit')
        mixed = cov.group_for_parameterisation(session, [wat, free])
        if len(mixed) != 1 or mixed[0]['seed'] is not free:
            _fail('water should be dropped, leaving only the free-ligand unit: %r' % (mixed,))
        print('PASS: solvent (HOH) excluded as a parameterisation seed')
    finally:
        session.models.close([m])

    print('ALL PASS')


# ChimeraX --script provides `session` in the module globals.
try:
    session  # noqa: F821
except NameError:
    session = None
if session is not None:
    run(session)
