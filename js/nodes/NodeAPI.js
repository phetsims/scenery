// Copyright 2020, University of Colorado Boulder

/**
 * PhET-iO API type for PressListener. Since PressListener is not an instrumented PhetioObject, but rather holds
 * instrumented sub-components, this API does not extend PhetioObjectAPI.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import Property from '../../../axon/js/Property.js';
import PropertyAPI from '../../../axon/js/PropertyAPI.js';
import merge from '../../../phet-core/js/merge.js';
import PhetioObjectAPI from '../../../tandem/js/PhetioObjectAPI.js';
import BooleanIO from '../../../tandem/js/types/BooleanIO.js';
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

      inputEnabledPropertyPhetioInstrumented: false,
      inputEnabledPropertyOptions: {
        phetioType: Property.PropertyIO( BooleanIO ),
        phetioFeatured: true
      },

      enabledPropertyPhetioInstrumented: false,
      enabledPropertyOptions: {
        phetioFeatured: true,
        phetioType: Property.PropertyIO( BooleanIO )
      }
    }, options );
    super( options );

    // @public (read-only)
    this.visibleProperty = new PropertyAPI( options.visiblePropertyOptions );

    if ( options.inputEnabledPropertyPhetioInstrumented ) {

      // @public (read-only)
      this.inputEnabledProperty = new PropertyAPI( options.inputEnabledPropertyOptions );
    }

    if ( options.enabledPropertyPhetioInstrumented ) {

      // @public (read-only)
      this.enabledProperty = new PropertyAPI( options.enabledPropertyOptions );
    }

    // TODO: not supported yet, see https://github.com/phetsims/scenery/issues/1098
    if ( options.opacityPropertyInstrumented ) {

      // @public (read-only)
      this.opacityProperty = new PropertyAPI( options.opacityPropertyOptions );
    }
  }
}

scenery.register( 'NodeAPI', NodeAPI );
export default NodeAPI;