// Copyright 2016, University of Colorado Boulder

/**
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var scenery = require( 'SCENERY/scenery' );

  // phet-io modules
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertions/assertInstanceOf' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );
  var TBoolean = require( 'ifphetio!PHET_IO/types/TBoolean' );
  var TNumber = require( 'ifphetio!PHET_IO/types/TNumber' );
  var TObject = require( 'ifphetio!PHET_IO/types/TObject' );
  var TVoid = require( 'ifphetio!PHET_IO/types/TVoid' );
  var TFunctionWrapper = require( 'ifphetio!PHET_IO/types/TFunctionWrapper' );

  /**
   * Wrapper type for phet/scenery's Node
   * @param node
   * @param phetioID
   * @constructor
   */
  function TNode( node, phetioID ) {
    TObject.call( this, node, phetioID );
    assertInstanceOf( node, phet.scenery.Node );
  }

  phetioInherit( TObject, 'TNode', TNode, {

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

    addPickableListener: {
      returnType: TVoid,
      parameterTypes: [ TFunctionWrapper( TVoid, [ TBoolean ] ) ],
      implementation: function( callback ) {
        var inst = this.instance;
        this.instance.on( 'pickability', function() {
          callback( inst.isPickable() );
        } );
      },
      documentation: 'Adds a listener for when pickability of the node changes'
    },

    addVisibleListener: {
      returnType: TVoid,
      parameterTypes: [ TFunctionWrapper( TVoid, [ TBoolean ] ) ],
      implementation: function( callback ) {
        var inst = this.instance;
        this.instance.on( 'visibility', function() {
          callback( inst.isVisible() );
        } );
      },
      documentation: 'Adds a listener for when visibility of the node changes'
    },

    setOpacity: {
      returnType: TVoid,
      parameterTypes: [ TNumber() ],
      implementation: function( opacity ) {
        this.instance.opacity = opacity;
      },
      documentation: 'Set opacity between 0-1 (inclusive)'
    },

    setRotation: {
      returnType: TVoid,
      parameterTypes: [ TNumber() ],
      implementation: function( rotation ) {
        this.instance.rotation = rotation;
      },
      documentation: 'Set the rotation of the node, in radians'
    }
  }, {
    documentation: 'The base type for graphical and potentially interactive objects'
  } );

  scenery.register( 'TNode', TNode );

  return TNode;
} );

