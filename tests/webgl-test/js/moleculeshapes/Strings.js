// Copyright 2002-2012, University of Colorado

var phet = phet || {};
phet.moleculeshapes = phet.moleculeshapes || {};

(function () {
    var englishStrings = {
        "molecule-shapes.name": "Molecule Shapes",
        "molecule-shapes.title": "Model",
        "molecule-shapes-basics.name": "Molecule Shapes: Basics",
        "molecule-shapes-basics.title": "Model",
        "real-molecules": "Real Molecules",

        "geometry.empty": "",
        "geometry.diatomic": "Linear",
        "geometry.linear": "Linear",
        "geometry.trigonalPlanar": "Trigonal Planar",
        "geometry.tetrahedral": "Tetrahedral",
        "geometry.trigonalBipyramidal": "Trigonal Bipyramidal",
        "geometry.octahedral": "Octahedral",

        "shape.empty": "",
        "shape.diatomic": "Linear",
        "shape.linear": "Linear",
        "shape.bent": "Bent",
        "shape.trigonalPlanar": "Trigonal Planar",
        "shape.trigonalPyramidal": "Trigonal Pyramidal",
        "shape.tShaped": "T-shaped",
        "shape.tetrahedral": "Tetrahedral",
        "shape.seesaw": "Seesaw",
        "shape.squarePlanar": "Square Planar",
        "shape.trigonalBipyramidal": "Trigonal Bipyramidal",
        "shape.squarePyramidal": "Square Pyramidal",
        "shape.octahedral": "Octahedral",

        "control.bonding": "Bonding",
        "control.lonePair": "Lone Pair",
        "control.options": "Options",
        "control.geometryName": "Name",
        "control.moleculeGeometry": "Molecule Geometry",
        "control.electronGeometry": "Electron Geometry",

        "control.showLonePairs": "Show Lone Pairs",
        "control.showAllLonePairs": "Show Outer Lone Pairs",
        "control.showBondAngles": "Show Bond Angles",
        "control.removeAll": "Remove All",

        "control.molecule": "Molecule",
        "control.realView": "Real",
        "control.modelView": "Model",

        "realExamples.title": "Real Examples",

        "angle.degrees": "{0}\u00B0",
        "angle.greaterThanDegrees": "greater than {0}\u00B0",
        "angle.lessThanDegrees": "less than {0}\u00B0"
    };

    var strings = englishStrings;

    phet.moleculeshapes.strings = {
        ANGLE__DEGREES: strings[ "angle.degrees" ],
        ANGLE__GREATER_THAN_DEGREES: strings[ "angle.greaterThanDegrees" ],
        ANGLE__LESS_THAN_DEGREES: strings[ "angle.lessThanDegrees" ],
        CONTROL__BONDING: strings[ "control.bonding" ],
        CONTROL__ELECTRON_GEOMETRY: strings[ "control.electronGeometry" ],
        CONTROL__GEOMETRY_NAME: strings[ "control.geometryName" ],
        CONTROL__LONE_PAIR: strings[ "control.lonePair" ],
        CONTROL__MODEL_VIEW: strings[ "control.modelView" ],
        CONTROL__MOLECULE: strings[ "control.molecule" ],
        CONTROL__MOLECULE_GEOMETRY: strings[ "control.moleculeGeometry" ],
        CONTROL__OPTIONS: strings[ "control.options" ],
        CONTROL__REAL_VIEW: strings[ "control.realView" ],
        CONTROL__REMOVE_ALL: strings[ "control.removeAll" ],
        CONTROL__SHOW_ALL_LONE_PAIRS: strings[ "control.showAllLonePairs" ],
        CONTROL__SHOW_BOND_ANGLES: strings[ "control.showBondAngles" ],
        CONTROL__SHOW_LONE_PAIRS: strings[ "control.showLonePairs" ],
        GEOMETRY__DIATOMIC: strings[ "geometry.diatomic" ],
        GEOMETRY__EMPTY: strings[ "geometry.empty" ],
        GEOMETRY__LINEAR: strings[ "geometry.linear" ],
        GEOMETRY__OCTAHEDRAL: strings[ "geometry.octahedral" ],
        GEOMETRY__TETRAHEDRAL: strings[ "geometry.tetrahedral" ],
        GEOMETRY__TRIGONAL_BIPYRAMIDAL: strings[ "geometry.trigonalBipyramidal" ],
        GEOMETRY__TRIGONAL_PLANAR: strings[ "geometry.trigonalPlanar" ],
        MOLECULE__SHAPES__BASICS__TITLE: strings[ "molecule-shapes-basics.title" ],
        MOLECULE__SHAPES__TITLE: strings[ "molecule-shapes.title" ],
        REAL__MOLECULES: strings[ "real-molecules" ],
        REAL_EXAMPLES__TITLE: strings[ "realExamples.title" ],
        SHAPE__BENT: strings[ "shape.bent" ],
        SHAPE__DIATOMIC: strings[ "shape.diatomic" ],
        SHAPE__EMPTY: strings[ "shape.empty" ],
        SHAPE__LINEAR: strings[ "shape.linear" ],
        SHAPE__OCTAHEDRAL: strings[ "shape.octahedral" ],
        SHAPE__SEESAW: strings[ "shape.seesaw" ],
        SHAPE__SQUARE_PLANAR: strings[ "shape.squarePlanar" ],
        SHAPE__SQUARE_PYRAMIDAL: strings[ "shape.squarePyramidal" ],
        SHAPE__T_SHAPED: strings[ "shape.tShaped" ],
        SHAPE__TETRAHEDRAL: strings[ "shape.tetrahedral" ],
        SHAPE__TRIGONAL_BIPYRAMIDAL: strings[ "shape.trigonalBipyramidal" ],
        SHAPE__TRIGONAL_PLANAR: strings[ "shape.trigonalPlanar" ],
        SHAPE__TRIGONAL_PYRAMIDAL: strings[ "shape.trigonalPyramidal" ]
    };
})();


