// Copyright 2017-2020, University of Colorado Boulder

/**
 * IO Type for Node
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare (PhET Interactive Simulations)
 */

import NumberProperty from '../../../axon/js/NumberProperty.js';
import PropertyIO from '../../../axon/js/PropertyIO.js';
import Range from '../../../dot/js/Range.js';
import merge from '../../../phet-core/js/merge.js';
import BooleanIO from '../../../tandem/js/types/BooleanIO.js';
import IOType from '../../../tandem/js/types/IOType.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import scenery from '../scenery.js';
import NodeProperty from '../util/NodeProperty.js';

const NodeIO = new IOType( 'NodeIO', {
  valueType: scenery.Node,
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
                 'For more about Scenery node pickability, please see <a href="http://phetsims.github.io/scenery/doc/implementation-notes#pickability">http://phetsims.github.io/scenery/doc/implementation-notes#pickability</a>',

  // TODO: https://github.com/phetsims/tandem/issues/211 Move these added Properties to the core types
  createWrapper: ( node, phetioID ) => {

    const pickableProperty = new NodeProperty( node, node.pickableProperty, 'pickable', merge( {

      // pick the baseline value from the parent Node's baseline
      phetioReadOnly: node.phetioReadOnly,

      tandem: node.tandem.createTandem( 'pickableProperty' ),
      phetioType: PropertyIO( NullableIO( BooleanIO ) ),
      phetioDocumentation: 'Sets whether the node will be pickable (and hence interactive), see the NodeIO documentation for more details'
    }, node.phetioComponentOptions, node.phetioComponentOptions.pickableProperty ) );

    // Adapter for the opacity.  Cannot use NodeProperty at the moment because it doesn't handle numeric types
    // properly--we may address this by moving to a mixin pattern.
    const opacityProperty = new NumberProperty( node.opacity, merge( {

      // pick the baseline value from the parent Node's baseline
      phetioReadOnly: node.phetioReadOnly,

      tandem: node.tandem.createTandem( 'opacityProperty' ),
      range: new Range( 0, 1 ),
      phetioDocumentation: 'Opacity of the parent NodeIO, between 0 (invisible) and 1 (fully visible)'
    }, node.phetioComponentOptions, node.phetioComponentOptions.opacityProperty ) );
    opacityProperty.link( function( opacity ) { node.opacity = opacity; } );
    node.opacityProperty.lazyLink( () => { opacityProperty.value = node.opacity; } );

    return {
      phetioObject: node,
      phetioID: phetioID,
      dispose: () => {
        pickableProperty.dispose();
        opacityProperty.dispose();
      }
    };
  }
} );

scenery.register( 'NodeIO', NodeIO );
export default NodeIO;