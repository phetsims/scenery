// Copyright 2002-2012, University of Colorado

/**
 * Displays a 3D Molecule
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.moleculeshapes = phet.moleculeshapes || {};
phet.moleculeshapes.view = phet.moleculeshapes.view || {};

// create a new scope
(function () {

    var defaultAtomRadius = 2;
    var defaultBondRadius = defaultAtomRadius / 4;

    var colorWrapper = function ( red, green, blue, alpha ) {
        return {
            preRender: function ( args ) {
                args.gl.uniform4f( args.shaderProgram.atomColor, red, green, blue, alpha );
            },

            postRender: function ( args ) {

            }
        }
    };

    var whiteColorWrapper = colorWrapper( 1, 1, 1, 1 );
    var centerColorWrapper = colorWrapper( 159 / 255.0, 102 / 255.0, 218 / 255.0, 1 );

    phet.moleculeshapes.view.GLMoleculeNode = function ( gl, molecule ) {
        phet.webgl.GLNode.call( this );

        var moleculeNode = this;

        this.atomNodes = [];
        this.bondNodes = [];

        this.gl = gl;
        this.molecule = molecule;

        var atoms = molecule.getAtoms();
        for( var i = 0; i < atoms; i++ ) {
            this.addAtom( atoms[i] );
        }
        molecule.onGroupAdded.addListener( function(group){
            if( group.isLonePair) {

            } else {
                moleculeNode.addAtom( group );
            }
        });

//        var sphere1 = new phet.webgl.Sphere( gl, defaultAtomRadius, 25, 25 );
//        sphere1.wrappers.push( centerColorWrapper );
//
//        var bondDistance = 2;
//        var bondSq2 = bondDistance / Math.sqrt( 2 );
//
//        var sphere2 = new phet.webgl.Sphere( gl, defaultAtomRadius, 25, 25 );
//        sphere2.transform.append( phet.math.Matrix4.translation( -bondSq2, bondSq2, 0 ) );
//        sphere2.wrappers.push( whiteColorWrapper );
//
//        var sphere3 = new phet.webgl.Sphere( gl, defaultAtomRadius, 25, 25 );
//        sphere3.transform.append( phet.math.Matrix4.translation( bondSq2, -bondSq2, 0 ) );
//        sphere3.wrappers.push( whiteColorWrapper );
//
//        var cylinder = new phet.webgl.Cylinder( gl, defaultBondRadius, bondDistance * 2, 16, 1 );
//        cylinder.transform.append( phet.math.Matrix4.rotationZ( Math.PI / 4 ) );
//        cylinder.transform.append( phet.math.Matrix4.rotationX( Math.PI / 2 ) );
//        cylinder.wrappers.push( whiteColorWrapper );
//
//        this.addChild( sphere1 );
//        this.addChild( sphere2 );
//        this.addChild( sphere3 );
//        this.addChild( cylinder );
    };

    var GLMoleculeNode = phet.moleculeshapes.view.GLMoleculeNode;

    GLMoleculeNode.prototype = Object.create( phet.webgl.GLNode.prototype );
    GLMoleculeNode.prototype.constructor = GLMoleculeNode;

    GLMoleculeNode.prototype.addAtom = function ( atom ) {
        var sphere = new phet.webgl.Sphere( this.gl, defaultAtomRadius, 25, 25 );

        // if it's the center atom
        if ( atom.position.get().equals( phet.math.Vector3.ZERO ) ) {
            sphere.wrappers.push( centerColorWrapper );
        }
        else {
            sphere.wrappers.push( whiteColorWrapper );
        }

        atom.position.addObserver( function () {
            var position = atom.position.get();
            sphere.transform.set( phet.math.Matrix4.translation( position.x, position.y, position.z ) );
        }, true );

        this.addChild( sphere );
    };
})();