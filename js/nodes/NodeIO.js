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
  var BooleanProperty = require( 'AXON/BooleanProperty' );
  var scenery = require( 'SCENERY/scenery' );

  // phet-io modules
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );
  var BooleanIO = require( 'ifphetio!PHET_IO/types/BooleanIO' );
  var FunctionIO = require( 'ifphetio!PHET_IO/types/FunctionIO' );
  var NullableIO = require( 'ifphetio!PHET_IO/types/NullableIO' );
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

    // TODO: This is in-progress work to convert object properties to Axon Properties, see https://github.com/phetsims/phet-io/issues/1326
    var visibleProperty = new BooleanProperty( node.visible, {
      tandem: node.tandem.createTandem( 'visibleProperty' ),
      instanceDocumentation: 'Property that controls whether the Node will be visible (and interactive).'
    } );

    visibleProperty.link( function( visible ) {
      node.visible = visible;
    } );
    node.on( 'visibility', function() {
      visibleProperty.value = node.visible;
    } );

  }

  phetioInherit( ObjectIO, 'NodeIO', NodeIO, {
    setPickable: {
      returnType: VoidIO,
      parameterTypes: [ NullableIO( BooleanIO ) ],
      implementation: function( pickable ) {
        this.instance.pickable = pickable;
      },
      documentation: 'Set whether the node will be pickable (and hence interactive). Pickable can take one of three values:<br><ul>' +
                     '<li>null: pass-through behavior. Nodes with input listeners are pickable, but nodes without input listeners won\'t block events for nodes behind it.</li>' +
                     '<li>false: The node cannot be interacted with, and it blocks events for nodes behind it.</li>' +
                     '<li>true: The node can be interacted with (if it has an input listener).</li></ul>' +
                     'For more about Scenery node pickability, please see <a href="http://phetsims.github.io/scenery/doc/implementation-notes#pickability">http://phetsims.github.io/scenery/doc/implementation-notes#pickability</a>.'
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
        pickable: node.isPickable(),
        opacity: node.opacity
      };
    },
    fromStateObject: function( stateObject ) {
      return stateObject;
    },
    setValue: function( node, fromStateObject ) {
      assert && assertInstanceOf( node, phet.scenery.Node );
      node.pickable = fromStateObject.pickable;
      node.opacity = fromStateObject.opacity;
    },
    documentation: 'The base type for graphical and potentially interactive objects.'
  } );

  scenery.register( 'NodeIO', NodeIO );

  return NodeIO;
} );

