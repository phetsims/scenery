// Copyright 2020, University of Colorado Boulder

/**
 * PhET-iO API type for PressListener. Since PressListener is not an instrumented PhetioObject, but rather holds
 * instrumented sub-components, this API does not extend PhetioObjectAPI.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import NumberPropertyIO from '../../../axon/js/NumberPropertyIO.js';
import PropertyAPI from '../../../axon/js/PropertyAPI.js';
import PropertyIO from '../../../axon/js/PropertyIO.js';
import merge from '../../../phet-core/js/merge.js';
import PhetioObjectAPI from '../../../tandem/js/PhetioObjectAPI.js';
import BooleanIO from '../../../tandem/js/types/BooleanIO.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import scenery from '../scenery.js';
import NodeIO from './NodeIO.js';

class NodeAPI extends PhetioObjectAPI {

  /**
   * @param {Object} [options]
   */
  constructor( options ) {

    options = merge( {
      phetioType: NodeIO,
      phetioState: false,
      visiblePropertyOptions: {
        phetioType: PropertyIO( BooleanIO )
      },

      pickablePropertyOptions: {
        phetioType: PropertyIO( NullableIO( BooleanIO ) )
      },

      opacityPropertyOptions: {
        phetioType: NumberPropertyIO
      }
    }, options );
    super( options );

    this.visibleProperty = new PropertyAPI( options.visiblePropertyOptions );
    this.opacityProperty = new PropertyAPI( options.opacityPropertyOptions );
    this.pickableProperty = new PropertyAPI( options.pickablePropertyOptions );
  }
}

scenery.register( 'NodeAPI', NodeAPI );
export default NodeAPI;