"""Build current materials after completing the validation calculation pipeline.

Run ``make validation`` on the full saved project, or the full experiment before
this command. Corrected report generation requires results/validation/ tables.
"""

from build_validation_materials import main

if __name__ == "__main__":
    main()
