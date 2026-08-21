'''
The interactive "Unparameterised Residues" validation panel was retired in favour
of the "Unparameterized Components" panel (``new_section.py``), which supersedes
it. Only this small shared helper -- used by that panel to resolve an MD template
name to its CCD reference residue + description -- is retained from the original
module.
'''


def _get_ccd_template_and_name(session, tname):
    from chimerax.isolde.openmm.amberff.template_utils import template_name_to_ccd_name
    from chimerax import mmcif
    ccd_name, extra_info = template_name_to_ccd_name(tname)
    if ccd_name is None:
        return (None, '')
    try:
        tmpl = mmcif.find_template_residue(session, ccd_name)
    except ValueError:
        return (None, "No CCD template found")
    description = tmpl.description
    if extra_info is not None:
        description += ', ' + extra_info
    return (tmpl, description)
