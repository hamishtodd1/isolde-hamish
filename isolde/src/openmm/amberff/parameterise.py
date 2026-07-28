
supported_elements = ('C','N','O','S','P','H','F','Cl','Br','I')

def residue_qualifies(residue):
    unsupported_elements = set(residue.atoms.element_names).difference(set(supported_elements))
    return len(unsupported_elements==0)

def parameterise_ligand(session, residue, net_charge=None, charge_method='am1-bcc'):
    from chimerax.core.errors import UserError
    if len(residue.neighbors) != 0:
        raise UserError(f"Residue is covalently bound to {len(residue.neighbors)} other residues. "
            "ISOLDE currenly only supports parameterising new non-covalent ligands.")
    import numpy
    element_check = numpy.isin(residue.atoms.element_names, supported_elements)
    if not numpy.all(element_check):
        unsupported = residue.atoms[numpy.logical_not(element_check)]
        raise UserError('Automatic ligand parameterisation currently only supports the following elements:\n'
            f'{", ".join(supported_elements)}\n'
            f'Residue type {residue.name} contains the unsupported elements {", ".join(numpy.unique(unsupported.element_names))}.'
            'If you wish to work with this residue you will need to parameterise it using an external package.')
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))
    gaff2_parms = os.path.join(base_path, 'gaff2.dat')
    from chimerax.add_charge.charge import nonstd_charge, estimate_net_charge
    from chimerax.atomic import Residues
    if net_charge is None:
        # Workaround - trigger recalculation of idatm_types if necessary to make sure rings are current
        atom_types = residue.atoms.idatm_types
        net_charge = estimate_net_charge(residue.atoms)
    from chimerax.amber_info import amber_bin
    import tempfile
    with tempfile.TemporaryDirectory() as tempdir:
        nonstd_charge(session, Residues([residue]), net_charge, charge_method, temp_dir=tempdir)
        ante_out = os.path.join(tempdir, "ante.out.mol2")
        parmchk2 = os.path.join(amber_bin, 'parmchk2')
        frcmod_file = os.path.join(tempdir, "residue.frcmod")
        import subprocess
        subprocess.check_output([parmchk2, '-s','2','-i',ante_out,'-f','mol2','-p',gaff2_parms,'-o',frcmod_file])
        from .amber_convert import amber_to_ffxml
        try:
            amber_to_ffxml(frcmod_file, ante_out, output_name=residue.name+'.xml', atom_names_from=residue)
        except UserError:
            session.logger.warning("Failed to assign atom names in the template based on the residue. This shouldn't "
                                   "affect most use cases, but will make hand-editing of the template more difficult.")
            amber_to_ffxml(frcmod_file, ante_out, output_name=residue.name+'.xml', atom_names_from=None)
        import shutil
        shutil.copyfile(ante_out, f'{residue.name}.mol2')

def parameterise_cmd(session, residues, override=False, net_charge=None,
                     always_raise_errors=True, shell_radius=1, base_templates=None,
                     reference_model=None, fetch_reference=False):
    from chimerax.core.errors import UserError
    from chimerax.atomic import Residues
    from .covalent import (group_for_parameterisation, parameterise_metal_site,
                           parameterise_covalent_unit, parameterise_free_ligand)

    # Classify every selected residue into the unit the pipeline actually builds
    # -- metal site, covalent unit or free ligand -- via the shared grouping
    # function (the SAME one the validation panel uses, so routing can never
    # diverge). Grouping is detection-only (metal-first, so a metal never falls
    # through to GAFF2); the builds happen per descriptor below.
    forcefield = None
    if hasattr(session, 'isolde'):
        ff_name = session.isolde.sim_params.forcefield
        forcefield = session.isolde.forcefield_mgr[ff_name]
    groups = group_for_parameterisation(session, residues, max_heavy_atoms=250,
                                        forcefield=forcefield)

    free_seeds = []
    for g in groups:
        kind = g['kind']
        if g['too_big']:
            msg = ('The %s unit %r has %d heavy atoms, above the limit of 250. '
                   'AM1-BCC on a fragment this large is unlikely to converge; '
                   'parameterise it externally.'
                   % (kind, g['unit'], g['num_heavy_atoms']))
            if always_raise_errors:
                raise UserError(msg)
            session.logger.warning(msg)
            continue
        if kind == 'metal':
            if g['unit'] is None:
                # Metal-involved but no coordination site could be resolved (e.g.
                # no donors found). Surface that, rather than misrouting to GAFF2.
                if always_raise_errors:
                    raise UserError(g['error'])
                session.logger.warning(g['error'])
                continue
            if g.get('unsupported'):
                # An unsupported metal (no bundled LJ params, e.g. Mo): refuse up
                # front, before the slow AM1-BCC inside parameterise_metal_site.
                if always_raise_errors:
                    raise UserError(g['unsupported'])
                session.logger.warning(g['unsupported'])
                continue
            try:
                parameterise_metal_site(session, g['unit'], shell_radius=shell_radius,
                                        net_charge=net_charge,
                                        base_templates=base_templates,
                                        reference_model=reference_model,
                                        fetch_reference=fetch_reference)
            except Exception as e:
                if always_raise_errors:
                    raise UserError(str(e))
                session.logger.warning('Metal-site parameterisation of %r failed: %s'
                                       % (g['unit'], e))
        elif kind == 'covalent':
            try:
                parameterise_covalent_unit(session, g['unit'], shell_radius=shell_radius,
                                           net_charge=net_charge,
                                           base_templates=base_templates)
            except Exception as e:
                if always_raise_errors:
                    raise UserError(str(e))
                session.logger.warning('Covalent parameterisation of %r failed: %s'
                                       % (g['unit'], e))
        else:  # free ligand
            if g['had_neighbours']:
                # Has inter-residue bonds but no non-standard linkage (a plain
                # in-chain residue): not a covalent unit and not a free ligand.
                msg = ('Selected residue(s) have no non-standard covalent linkage '
                       'to parameterise. For a free ligand use "isolde '
                       'parameterise" instead.')
                if always_raise_errors:
                    raise UserError(msg)
                session.logger.warning(msg)
                continue
            free_seeds.append(g['seed'])

    free = Residues(free_seeds)
    if not len(free):
        return
    unique_residue_types = [free[free.names == name][0] for name in free.unique_names]
    for residue in unique_residue_types:
        if hasattr(session, 'isolde'):
            ff_name = session.isolde.sim_params.forcefield
            forcefield = session.isolde.forcefield_mgr[ff_name]
            ligand_db = session.isolde.forcefield_mgr.ligand_db(ff_name)
            from chimerax.isolde.openmm.openmm_interface import find_residue_templates
            from chimerax.atomic import Residues
            templates = find_residue_templates(Residues([residue]), forcefield, ligand_db=ligand_db, logger=session.logger)
            if len(templates):
                if not override:
                    raise UserError(f'Residue name {residue.name} already corresponds to template {templates[0]} in '
                        f'the {ff_name} forcefield. If you wish to replace that template, re-run this '
                        'command with override=True')
        try:
            # Free ligands now go through the RDKit pipeline (correct bond orders,
            # order-based atom-name round-trip, self-contained collision-proof
            # template) -- the same machinery as the covalent path.
            from .covalent import parameterise_free_ligand
            parameterise_free_ligand(session, residue, net_charge=net_charge,
                                     base_templates=base_templates)
        except Exception as e:
            if always_raise_errors:
                raise UserError(str(e))
            else:
                session.logger.warning(f'Parameterisation of {residue.name} failed with the following message:')
                session.logger.warning(str(e))
                continue

def register_isolde_param(logger):
    from chimerax.atomic import ResiduesArg, AtomicStructureArg
    from chimerax.core.commands import (CmdDesc, BoolArg, IntArg, StringArg, ListOf,
                                        register)
    desc = CmdDesc(
        required=[('residues', ResiduesArg)],
        keyword=[
            ('override', BoolArg),
            ('net_charge', IntArg),
            ('always_raise_errors', BoolArg),
            ('shell_radius', IntArg),
            ('base_templates', ListOf(StringArg)),
            ('reference_model', AtomicStructureArg),
            ('fetch_reference', BoolArg),
        ],
        synopsis=('Parameterise ligand(s) for use in ISOLDE. Supports most organic species. '
            'Covalent ligands, residue-residue crosslinks and main-chain modifications are '
            'built as a capped super-residue (select any residue of the unit). Metal sites '
            '(hemes, Zn/Mg/Mn/Ca/... coordination) are built as a bonded metal template with '
            'soft empirical coordination terms (select the metalloligand or metal). Hydrogens '
            'must be present and correct. baseTemplates takes a list of CCD ids, registered '
            'template names and/or SMILES strings whose chemistry is matched onto the ligand '
            'to fix bond orders/charges when perception is unreliable. For a polynuclear '
            'inorganic cluster whose idealised geometry is not built in, referenceModel takes '
            'an open high-quality structure containing the same ligand to use as the geometry '
            'exemplar; fetchReference true instead lets ISOLDE fetch the highest-resolution '
            'PDB structure containing it (a network lookup, off by default).')
    )
    register('isolde parameterise', desc, parameterise_cmd, logger=logger)
