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
  var Property = require( 'AXON/Property' );
  var PropertyIO = require( 'AXON/PropertyIO' );
  var scenery = require( 'SCENERY/scenery' );

  // phet-io modules
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );
  var BooleanIO = require( 'ifphetio!PHET_IO/types/BooleanIO' );
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
      phetioInstanceDocumentation: 'Property that controls whether the Node will be visible (and interactive), see the NodeIO documentation for more details.'
    } );

    visibleProperty.link( function( visible ) {
      node.visible = visible;
    } );
    node.on( 'visibility', function() {
      visibleProperty.value = node.visible;
    } );

    var pickableProperty = new Property( node.pickable, {
      tandem: node.tandem.createTandem( 'pickableProperty' ),
      phetioType: PropertyIO( NullableIO( BooleanIO ) ),
      phetioInstanceDocumentation: 'Set whether the node will be pickable (and hence interactive), see the NodeIO documentation for more details.'
    } );
    pickableProperty.link( function( pickable ) {
      node.pickable = pickable;
    } );
    node.on( 'pickability', function() {
      pickableProperty.value = node.pickable;
    } );

    // @private
    this.disposeNodeIO = function() {
      visibleProperty.dispose();
      pickableProperty.dispose();
    };
  }

  phetioInherit( ObjectIO, 'NodeIO', NodeIO, {

    /**
     * @public - called by PhetioObject when the wrapper is done
     */
    dispose: function() {
      this.disposeNodeIO();
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
    documentation: 'The base type for graphical and potentially interactive objects.  NodeIO has nested PropertyIO values' +
                   'for visibility, pickability and opacity.<br>' +
                   'Pickable can take one of three values:<br><ul>' +
                   '<li>null: pass-through behavior. Nodes with input listeners are pickable, but nodes without input listeners won\\\'t block events for nodes behind it.</li>\' +\n' +
                   '<li>false: The node cannot be interacted with, and it blocks events for nodes behind it.</li>\' +\n' +
                   '<li>true: The node can be interacted with (if it has an input listener).</li></ul>\' +\n' +
                   'For more about Scenery node pickability, please see <a href="http://phetsims.github.io/scenery/doc/implementation-notes#pickability">http://phetsims.github.io/scenery/doc/implementation-notes#pickability</a>.\'\n' +
                   '},'
  } );

  scenery.register( 'NodeIO', NodeIO );

  return NodeIO;
} );

