// Copyright 2016, University of Colorado Boulder

/**
 * IO type for Node
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var NullableIO = require( 'ifphetio!PHET_IO/types/NullableIO' );
  var scenery = require( 'SCENERY/scenery' );

  // phet-io modules
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );
  var BooleanIO = require( 'ifphetio!PHET_IO/types/BooleanIO' );
  var FunctionIO = require( 'ifphetio!PHET_IO/types/FunctionIO' );
  var NumberIO = require( 'ifphetio!PHET_IO/types/NumberIO' );
  var ObjectIO = require( 'ifphetio!PHET_IO/types/ObjectIO' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );
  var VoidIO = require( 'ifphetio!PHET_IO/types/VoidIO' );

  /**
   * IO type for phet/scenery's Node
   * @param {Node} node
   * @param {string} phetioID
   * @constructor
   */
  function NodeIO( node, phetioID ) {
    assert && assertInstanceOf( node, phet.scenery.Node );
    ObjectIO.call( this, node, phetioID );
  }

  phetioInherit( ObjectIO, 'NodeIO', NodeIO, {

    detach: {
      returnType: VoidIO,
      parameterTypes: [],
      implementation: function() {
        this.instance.detach();
      },
      documentation: 'Detaches the node from its parents (if any)'
    },
    isVisible: {
      returnType: BooleanIO,
      parameterTypes: [],
      implementation: function() {
        return this.instance.visible;
      },
      documentation: 'Gets a Boolean value indicating whether the node can be seen and interacted with'
    },

    setVisible: {
      returnType: VoidIO,
      parameterTypes: [ BooleanIO ],
      implementation: function( visible ) {
        this.instance.visible = visible;
      },
      documentation: 'Set whether the node will be visible (and interactive)'
    },

    setPickable: {
      returnType: VoidIO,
      parameterTypes: [ NullableIO( BooleanIO ) ],
      implementation: function( pickable ) {
        this.instance.pickable = pickable;
      },
      documentation: 'Set whether the node will be pickable (and hence interactive)'
    },

    isPickable: {
      returnType: BooleanIO,
      parameterTypes: [],
      implementation: function() {
        return this.instance.pickable;
      },
      documentation: 'Gets whether the node is pickable (and hence interactive)'
    },

    addPickableListener: {
      returnType: VoidIO,
      parameterTypes: [ FunctionIO( VoidIO, [ BooleanIO ] ) ],
      implementation: function( callback ) {
        var inst = this.instance;
        this.instance.on( 'pickability', function() {
          callback( inst.isPickable() );
        } );
      },
      documentation: 'Adds a listener for when pickability of the node changes'
    },

    addVisibleListener: {
      returnType: VoidIO,
      parameterTypes: [ FunctionIO( VoidIO, [ BooleanIO ] ) ],
      implementation: function( callback ) {
        var inst = this.instance;
        this.instance.on( 'visibility', function() {
          callback( inst.isVisible() );
        } );
      },
      documentation: 'Adds a listener for when visibility of the node changes'
    },

    setOpacity: {
      returnType: VoidIO,
      parameterTypes: [ NumberIO ],
      implementation: function( opacity ) {
        this.instance.opacity = opacity;
      },
      documentation: 'Set opacity between 0-1 (inclusive)'
    },

    setRotation: {
      returnType: VoidIO,
      parameterTypes: [ NumberIO ],
      implementation: function( rotation ) {
        this.instance.rotation = rotation;
      },
      documentation: 'Set the rotation of the node, in radians'
    }
  }, {
    toStateObject: function( node ) {
      assert && assertInstanceOf( node, phet.scenery.Node );
      return {
        visible: node.isVisible(),
        pickable: node.isPickable(),
        opacity: node.opacity
      };
    },
    fromStateObject: function( stateObject ) {
      return stateObject;
    },
    setValue: function( node, fromStateObject ) {
      assert && assertInstanceOf( node, phet.scenery.Node );
      node.visible = fromStateObject.visible;
      node.pickable = fromStateObject.pickable;
      node.opacity = fromStateObject.opacity;
    },
    documentation: 'The base type for graphical and potentially interactive objects'
  } );

  scenery.register( 'NodeIO', NodeIO );

  return NodeIO;
} );

