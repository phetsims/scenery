// Copyright 2016, University of Colorado Boulder

/**
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare
 */
define( function( require ) {
  'use strict';

  // modules
  var assertInstanceOf = require( 'PHET_IO/assertions/assertInstanceOf' );
  var phetioInherit = require( 'PHET_IO/phetioInherit' );
  var phetioNamespace = require( 'PHET_IO/phetioNamespace' );
  var TBoolean = require( 'PHET_IO/types/TBoolean' );
  var TNumber = require( 'PHET_IO/types/TNumber' );
  var TObject = require( 'PHET_IO/types/TObject' );
  var TProperty = require( 'PHET_IO/types/axon/TProperty' );
  var TVoid = require( 'PHET_IO/types/TVoid' );

  var TNode = phetioInherit( TObject, 'TNode', function( node, phetioID ) {
    TObject.call( this, node, phetioID );
    assertInstanceOf( node, phet.scenery.Node );
  }, {

    // TODO: Not working yet at all.
    getVisibleProperty: {
      // Convert the visibility property to property interface.
      // TODO: Do this in scenery so adapters are not necessary here.
      returnType: TProperty( TBoolean ),
      parameterTypes: [],
      implementation: function() {

        // TODO: We need some way to unlink this.
        var property = new phet.axon.Property( this.instance.visible );
        return new TProperty( TBoolean )( property, null );
      },
      documentation: 'Gets a TProperty indicating whether the node is visible or not.'
    },
    isVisible: {
      returnType: TBoolean,
      parameterTypes: [],
      implementation: function() {
        return this.instance.visible;
      },
      documentation: 'Gets a Boolean value indicating whether the node can be seen and interacted with'
    },
    setVisible: {
      returnType: TVoid,
      parameterTypes: [ TBoolean ],
      implementation: function( visible ) {
        this.instance.visible = visible;
      },
      documentation: 'Set whether the node will be visible (and interactive)'
    },
    setPickable: {
      returnType: TVoid,
      parameterTypes: [ TBoolean ],
      implementation: function( pickable ) {
        this.instance.pickable = pickable;
      },
      documentation: 'Set whether the node will be pickable (and hence interactive)'
    },
    isPickable: {
      returnType: TBoolean,
      parameterTypes: [],
      implementation: function() {
        return this.instance.pickable;
      },
      documentation: 'Gets whether the node is pickable (and hence interactive)'
    },
    setOpacity: {
      returnType: TVoid,
      parameterTypes: [ TNumber ],
      implementation: function( opacity ) {
        this.instance.opacity = opacity;
      },
      documentation: 'Set opacity between 0-1 (inclusive)'
    },
    setRotation: {
      returnType: TVoid,
      parameterTypes: [ TNumber ],
      implementation: function( rotation ) {
        this.instance.rotation = rotation;
      },
      documentation: 'Set the rotation of the node, in radians'
    }
  }, {
    documentation: 'The base type for graphical and potentially interactive objects'
  } );

  phetioNamespace.register( 'TNode', TNode );

  return TNode;
} );

