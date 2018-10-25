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
  var NodeProperty = require( 'SCENERY/util/NodeProperty' );
  var NumberProperty = require( 'AXON/NumberProperty' );
  var phetioInherit = require( 'TANDEM/phetioInherit' );
  var PropertyIO = require( 'AXON/PropertyIO' );
  var scenery = require( 'SCENERY/scenery' );
  var BooleanIO = require( 'TANDEM/types/BooleanIO' );
  var NullableIO = require( 'TANDEM/types/NullableIO' );
  var ObjectIO = require( 'TANDEM/types/ObjectIO' );

  // ifphetio
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );
  var Range = require( 'DOT/Range' );

  /**
   * IO type for phet/scenery's Node
   * @param {Node} node
   * @param {string} phetioID
   * @constructor
   */
  function NodeIO( node, phetioID ) {
    assert && assertInstanceOf( node, scenery.Node );
    ObjectIO.call( this, node, phetioID );

    // TODO: This is in-progress work to convert object properties to Axon Properties, see https://github.com/phetsims/phet-io/issues/1326
    var visibleProperty = new NodeProperty( node, 'visibility', 'visible', {

      // pick the following values from the parent Node
      phetioReadOnly: node.phetioReadOnly,
      phetioState: node.phetioState,
      phetioType: PropertyIO( BooleanIO ),

      tandem: node.tandem.createTandem( 'visibleProperty' ),
      phetioDocumentation: 'Controls whether the Node will be visible (and interactive), see the NodeIO documentation for more details.'
    } );

    var pickableProperty = new NodeProperty( node, 'pickability', 'pickable', {

      // pick the following values from the parent Node
      phetioReadOnly: node.phetioReadOnly,
      phetioState: node.phetioState,

      tandem: node.tandem.createTandem( 'pickableProperty' ),
      phetioType: PropertyIO( NullableIO( BooleanIO ) ),
      phetioDocumentation: 'Sets whether the node will be pickable (and hence interactive), see the NodeIO documentation for more details'
    } );

    // Adapter for the opacity.  Cannot use NodeProperty at the moment because it doesn't handle numeric types
    // properly--we may address this by moving to a mixin pattern.
    var opacityProperty = new NumberProperty( node.opacity, {

      // pick the following values from the parent Node
      phetioReadOnly: node.phetioReadOnly,
      phetioState: node.phetioState,

      tandem: node.tandem.createTandem( 'opacityProperty' ),
      range: new Range( 0, 1 ),
      phetioDocumentation: 'Opacity of the parent NodeIO, between 0 (invisible) and 1 (fully visible)'
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
     * We can't declare the Node phetioState: false, like the general pattern that we want to use ( see, because that property
     * bubbles to children, and we want things like visibleProperty to be in the state by default, just not the NodeIO
     * type itself.
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
      return o; // Pass through values defined by subclasses
    },

    documentation: 'The base type for graphical and potentially interactive objects.  NodeIO has nested PropertyIO values ' +
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

