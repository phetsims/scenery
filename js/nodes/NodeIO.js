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
  var NumberProperty = require( 'AXON/NumberProperty' );
  var Property = require( 'AXON/Property' );
  var PropertyIO = require( 'AXON/PropertyIO' );
  var scenery = require( 'SCENERY/scenery' );

  // phet-io modules
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );
  var BooleanIO = require( 'ifphetio!PHET_IO/types/BooleanIO' );
  var NullableIO = require( 'ifphetio!PHET_IO/types/NullableIO' );
  var ObjectIO = require( 'ifphetio!PHET_IO/types/ObjectIO' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );

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

    visibleProperty.link( function( visible ) { node.visible = visible; } );
    node.on( 'visibility', function() { visibleProperty.value = node.visible; } );

    var pickableProperty = new Property( node.pickable, {
      tandem: node.tandem.createTandem( 'pickableProperty' ),
      phetioType: PropertyIO( NullableIO( BooleanIO ) ),
      phetioInstanceDocumentation: 'Set whether the node will be pickable (and hence interactive), see the NodeIO documentation for more details.'
    } );
    pickableProperty.link( function( pickable ) { node.pickable = pickable; } );
    node.on( 'pickability', function() { pickableProperty.value = node.pickable; } );

    var opacityProperty = new NumberProperty( node.opacity, {
      tandem: node.tandem.createTandem( 'opacityProperty' ),
      range: { min: 0, max: 1 },
      phetioInstanceDocumentation: 'Opacity of the parent NodeIO, between 0 (invisible) and 1 (fully visible).'
    } );
    opacityProperty.link( function( opacity ) { node.opacity = opacity; } );
    node.on( 'opacity', function() { opacityProperty.value = node.opacity; } );

    // @private
    this.disposeNodeIO = function() {
      visibleProperty.dispose();
      pickableProperty.dispose();
      opacityProperty.dispose();
    };
  }

  phetioInherit( ObjectIO, 'NodeIO', NodeIO, {

    /**
     * @public - called by PhetioObject when the wrapper is done
     */
    dispose: function() {
      this.disposeNodeIO();
    }
  }, {

    /**
     * Since NodeIO has no intrinsic state, we must signify to the PhET-iO serialization engine that this Node should
     * opt out of appearing in the state. We can't achieve this by removing this method, since NodeIO's parent type has
     * a `toStateObject` that causes a cyclical JSON reference.
     *
     * We also don't need matching `fromStateObject` and `setValue` methods because this type won't be added to the state
     * object, and thus setState will never be called on instances of NodeIO.
     *
     * Subtypes can still override this method, and implement their own `fromStateObject` and `setValue` if there is desired
     * serializable data for that type to hold in the state.
     *
     * @returns {undefined} - We don't use null because other types want that value in the state, see `NullableIO` for example.
     * @override
     */
    toStateObject: function() {
      return undefined;
    },

    /**
     * @param {Node} o
     * @returns {Object}
     * @override - to prevent attempted JSON serialization of circular Node
     */
    fromStateObject: function( o ) {
      return o; // Pass through values defined by subclasses, such as AquaRadioButtonIO.enabled
    },

    documentation: 'The base type for graphical and potentially interactive objects.  NodeIO has nested PropertyIO values' +
                   'for visibility, pickability and opacity.' +
                   '<br>' +
                   '<br>' +
                   'Pickable can take one of three values:<br>' +
                   '<ul>' +
                   '<li>null: pass-through behavior. Nodes with input listeners are pickable, but nodes without input listeners won\'t block events for nodes behind it.</li>' +
                   '<li>false: The node cannot be interacted with, and it blocks events for nodes behind it.</li>' +
                   '<li>true: The node can be interacted with (if it has an input listener).</li>' +
                   '</ul>' +
                   'For more about Scenery node pickability, please see <a href="http://phetsims.github.io/scenery/doc/implementation-notes#pickability">http://phetsims.github.io/scenery/doc/implementation-notes#pickability</a>'
  } );

  scenery.register( 'NodeIO', NodeIO );

  return NodeIO;
} );
