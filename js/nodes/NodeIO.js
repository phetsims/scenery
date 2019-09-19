// Copyright 2017-2019, University of Colorado Boulder

/**
 * IO type for Node
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare (PhET Interactive Simulations)
 */
define( require => {
  'use strict';

  // modules
  const BooleanIO = require( 'TANDEM/types/BooleanIO' );
  const NodeProperty = require( 'SCENERY/util/NodeProperty' );
  const NullableIO = require( 'TANDEM/types/NullableIO' );
  const NumberProperty = require( 'AXON/NumberProperty' );
  const ObjectIO = require( 'TANDEM/types/ObjectIO' );
  const PropertyIO = require( 'AXON/PropertyIO' );
  const Range = require( 'DOT/Range' );
  const scenery = require( 'SCENERY/scenery' );

  class NodeIO extends ObjectIO {
    constructor( node, phetioID ) {
      super( node, phetioID );

      // TODO: This is in-progress work to convert object properties to Axon Properties, see https://github.com/phetsims/phet-io/issues/1326
      var visibleProperty = new NodeProperty( node, 'visibility', 'visible', _.extend( {

        // pick the baseline value from the parent Node's baseline
        phetioReadOnly: node.phetioReadOnly,
        phetioType: PropertyIO( BooleanIO ),

        tandem: node.tandem.createTandem( 'visibleProperty' ),
        phetioDocumentation: 'Controls whether the Node will be visible (and interactive), see the NodeIO documentation for more details.'
      }, node.phetioComponentOptions, node.phetioComponentOptions.visibleProperty ) );

      var pickableProperty = new NodeProperty( node, 'pickability', 'pickable', _.extend( {

        // pick the baseline value from the parent Node's baseline
        phetioReadOnly: node.phetioReadOnly,

        tandem: node.tandem.createTandem( 'pickableProperty' ),
        phetioType: PropertyIO( NullableIO( BooleanIO ) ),
        phetioDocumentation: 'Sets whether the node will be pickable (and hence interactive), see the NodeIO documentation for more details'
      }, node.phetioComponentOptions, node.phetioComponentOptions.pickableProperty ) );

      // Adapter for the opacity.  Cannot use NodeProperty at the moment because it doesn't handle numeric types
      // properly--we may address this by moving to a mixin pattern.
      var opacityProperty = new NumberProperty( node.opacity, _.extend( {

        // pick the baseline value from the parent Node's baseline
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

    /**
     * @public - called by PhetioObject when the wrapper is done
     */
    dispose() {
      this.disposeNodeIO();
    }
  }

  NodeIO.validator = { valueType: scenery.Node };

  NodeIO.documentation = 'The base type for graphical and potentially interactive objects.  NodeIO has nested PropertyIO values ' +
                         'for visibility, pickability and opacity.' +
                         '<br>' +
                         '<br>' +
                         'Pickable can take one of three values:<br>' +
                         '<ul>' +
                         '<li>null: pass-through behavior. Nodes with input listeners are pickable, but nodes without input listeners won\'t block events for nodes behind it.</li>' +
                         '<li>false: The node cannot be interacted with, and it blocks events for nodes behind it.</li>' +
                         '<li>true: The node can be interacted with (if it has an input listener).</li>' +
                         '</ul>' +
                         'For more about Scenery node pickability, please see <a href="http://phetsims.github.io/scenery/doc/implementation-notes#pickability">http://phetsims.github.io/scenery/doc/implementation-notes#pickability</a>';
  NodeIO.typeName = 'NodeIO';
  ObjectIO.validateSubtype( NodeIO );

  return scenery.register( 'NodeIO', NodeIO );
} );

