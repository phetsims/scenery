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
  var BooleanIO = require( 'TANDEM/types/BooleanIO' );
  var NodeProperty = require( 'SCENERY/util/NodeProperty' );
  var NullableIO = require( 'TANDEM/types/NullableIO' );
  var NumberProperty = require( 'AXON/NumberProperty' );
  var ObjectIO = require( 'TANDEM/types/ObjectIO' );
  var phetioInherit = require( 'TANDEM/phetioInherit' );
  var PropertyIO = require( 'AXON/PropertyIO' );
  var Range = require( 'DOT/Range' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * IO type for phet/scenery's Node
   * @param {Node} node
   * @param {string} phetioID
   * @constructor
   */
  function NodeIO( node, phetioID ) {
    ObjectIO.call( this, node, phetioID );

    // TODO: This is in-progress work to convert object properties to Axon Properties, see https://github.com/phetsims/phet-io/issues/1326
    var visibleProperty = new NodeProperty( node, 'visibility', 'visible', _.extend( {

      // pick the following values from the parent Node
      phetioReadOnly: node.phetioReadOnly,
      phetioType: PropertyIO( BooleanIO ),

      tandem: node.tandem.createTandem( 'visibleProperty' ),
      phetioDocumentation: 'Controls whether the Node will be visible (and interactive), see the NodeIO documentation for more details.'
    }, node.phetioComponentOptions, node.phetioComponentOptions.visibleProperty ) );

    var pickableProperty = new NodeProperty( node, 'pickability', 'pickable', _.extend( {

      // pick the following values from the parent Node
      phetioReadOnly: node.phetioReadOnly,

      tandem: node.tandem.createTandem( 'pickableProperty' ),
      phetioType: PropertyIO( NullableIO( BooleanIO ) ),
      phetioDocumentation: 'Sets whether the node will be pickable (and hence interactive), see the NodeIO documentation for more details'
    }, node.phetioComponentOptions, node.phetioComponentOptions.pickableProperty ) );

    // Adapter for the opacity.  Cannot use NodeProperty at the moment because it doesn't handle numeric types
    // properly--we may address this by moving to a mixin pattern.
    var opacityProperty = new NumberProperty( node.opacity, _.extend( {

      // pick the following values from the parent Node
      phetioReadOnly: node.phetioReadOnly,

      tandem: node.tandem.createTandem( 'opacityProperty' ),
      range: new Range( 0, 1 ),
      phetioDocumentation: 'Opacity of the parent NodeIO, between 0 (invisible) and 1 (fully visible)'
    }, node.phetioComponentOptions, node.phetioComponentOptions.opacityProperty ) );
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

    validator: { valueType: scenery.Node },

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

