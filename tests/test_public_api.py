import hyperspatial as hs

def test_public_concept_objects_are_exported():
    for name in ('PanelDesign','AdaptationResult','GeneResult','ForecastResult','RunManifest'):
        assert hasattr(hs, name)

def test_cli_has_release_commands():
    from hyperspatial.cli import parser
    help_text = parser().format_help()
    for command in ('design','adapt','forecast','validate','report'):
        assert command in help_text

