// Copyright 2020, University of Colorado Boulder

/**
 * PhET-iO API type for PressListener. Since PressListener is not an instrumented PhetioObject, but rather holds
 * instrumented sub-components, this API does not extend PhetioObjectAPI.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import PropertyAPI from '../../../axon/js/PropertyAPI.js';
import Property from '../../../axon/js/Property.js';
import merge from '../../../phet-core/js/merge.js';
import PhetioObjectAPI from '../../../tandem/js/PhetioObjectAPI.js';
import BooleanIO from '../../../tandem/js/types/BooleanIO.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import scenery from '../scenery.js';
import Node from './Node.js';

class NodeAPI extends PhetioObjectAPI {

  /**
   * @param {Object} [options]
   */
  constructor( options ) {

    options = merge( {
      phetioType: Node.NodeIO,
      phetioState: false,
      visiblePropertyOptions: {
        phetioType: Property.PropertyIO( BooleanIO )
      },

      pickablePropertyInstrumented: false,
      pickablePropertyOptions: {
        phetioType: Property.PropertyIO( NullableIO( BooleanIO ) )
      }
    }, options );
    super( options );

    this.visibleProperty = new PropertyAPI( options.visiblePropertyOptions );

    if ( options.pickablePropertyInstrumented ) {
      this.pickableProperty = new PropertyAPI( options.pickablePropertyOptions );
    }

    // TODO: not supported yet, see https://github.com/phetsims/scenery/issues/1098
    if ( options.opacityPropertyInstrumented ) {
      this.opacityProperty = new PropertyAPI( options.opacityPropertyOptions );
    }
  }
}

scenery.register( 'NodeAPI', NodeAPI );
export default NodeAPI;