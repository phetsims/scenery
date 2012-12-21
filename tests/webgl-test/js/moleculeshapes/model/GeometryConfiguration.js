// Copyright 2002-2012, University of Colorado

/**
 * Contains the "optimal" molecule structures (pair group directions stored as unit vectors),
 * in an order such that higher-repulsion pair groups (lone pairs)
 * will tend to occupy the 1st slots, and bonds will occupy the later slots.
 */

var phet = phet || {};
phet.moleculeshapes = phet.moleculeshapes || {};
phet.moleculeshapes.model = phet.moleculeshapes.model || {};

// create a new scope
(function () {
    var TETRA_CONST = Math.PI * -19.471220333 / 180;

    phet.moleculeshapes.model.GeometryConfiguration = function ( name, unitVectors ) {
        this.name = name;
        this.unitVectors = unitVectors;
    };

    var GeometryConfiguration = phet.moleculeshapes.model.GeometryConfiguration;

    var Strings = phet.moleculeshapes.strings;
    var Vector3 = phet.math.Vector3;

    var geometries = {
        0: new GeometryConfiguration( Strings.GEOMETRY__EMPTY, [] ),

        1: new GeometryConfiguration(
                Strings.GEOMETRY__DIATOMIC,
                [
                    new Vector3( 1, 0, 0 )
                ]
        ),
        2: new GeometryConfiguration(
                Strings.GEOMETRY__LINEAR,
                [
                    new Vector3( 1, 0, 0 ),
                    new Vector3( -1, 0, 0 )
                ]
        ),
        3: new GeometryConfiguration(
                Strings.GEOMETRY__TRIGONAL_PLANAR,
                [
                    new Vector3( 1, 0, 0 ),
                    new Vector3( Math.cos( Math.PI * 2 / 3 ), Math.sin( Math.PI * 2 / 3 ), 0 ),
                    new Vector3( Math.cos( Math.PI * 4 / 3 ), Math.sin( Math.PI * 4 / 3 ), 0 )
                ]
        ),
        4: new GeometryConfiguration(
                Strings.GEOMETRY__TETRAHEDRAL,
                [
                    new Vector3( 0, 0, 1 ),
                    new Vector3( Math.cos( 0 ) * Math.cos( TETRA_CONST ), Math.sin( 0 ) * Math.cos( TETRA_CONST ), Math.sin( TETRA_CONST ) ),
                    new Vector3( Math.cos( Math.PI * 2 / 3 ) * Math.cos( TETRA_CONST ), Math.sin( Math.PI * 2 / 3 ) * Math.cos( TETRA_CONST ), Math.sin( TETRA_CONST ) ),
                    new Vector3( Math.cos( Math.PI * 4 / 3 ) * Math.cos( TETRA_CONST ), Math.sin( Math.PI * 4 / 3 ) * Math.cos( TETRA_CONST ), Math.sin( TETRA_CONST ) )
                ]
        ),
        5: new GeometryConfiguration(
                Strings.GEOMETRY__TRIGONAL_BIPYRAMIDAL,
                [
                    // equitorial (fills up with lone pairs first)
                    new Vector3( 0, 1, 0 ),
                    new Vector3( 0, Math.cos( Math.PI * 2 / 3 ), Math.sin( Math.PI * 2 / 3 ) ),
                    new Vector3( 0, Math.cos( Math.PI * 4 / 3 ), Math.sin( Math.PI * 4 / 3 ) ),

                    // axial
                    new Vector3( 1, 0, 0 ),
                    new Vector3( -1, 0, 0 )
                ]
        ),
        6: new GeometryConfiguration(
                Strings.GEOMETRY__OCTAHEDRAL,
                [
                    // opposites first
                    new Vector3( 0, 0, 1 ),
                    new Vector3( 0, 0, -1 ),
                    new Vector3( 0, 1, 0 ),
                    new Vector3( 0, -1, 0 ),
                    new Vector3( 1, 0, 0 ),
                    new Vector3( -1, 0, 0 )
                ]
        )
    };

    GeometryConfiguration.getConfiguration = function ( numberOfGroups ) {
        return geometries[numberOfGroups];
    };
})();
