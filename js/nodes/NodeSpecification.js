// Copyright 2020, University of Colorado Boulder

import Property from '../../../axon/js/Property.js';
import merge from '../../../phet-core/js/merge.js';
import PhetioObjectSpecification from '../../../tandem/js/PhetioObjectSpecification.js';
import BooleanIO from '../../../tandem/js/types/BooleanIO.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import scenery from '../scenery.js';
import Node from './Node.js';

class NodeSpecification extends PhetioObjectSpecification {

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

      pickablePropertyPhetioInstrumented: false,
      pickablePropertyOptions: {
        phetioType: Property.PropertyIO( NullableIO( BooleanIO ) ),
        phetioFeatured: true
      },

      enabledPropertyPhetioInstrumented: false,
      enabledPropertyOptions: {
        phetioFeatured: true,
        phetioType: Property.PropertyIO( BooleanIO )
      }
    }, options );

    super( options );


    // TODO: https://github.com/phetsims/phet-io/issues/1657 Add these back in
    // @public (read-only)

    // this.visibleProperty = new PropertyAPI( options.visiblePropertyOptions );
    //
    // if ( options.pickablePropertyPhetioInstrumented ) {
    //
    //   // @public (read-only)
    //   this.pickableProperty = new PropertyAPI( options.pickablePropertyOptions );
    // }
    //
    // if ( options.enabledPropertyPhetioInstrumented ) {
    //
    //   // @public (read-only)
    //   this.enabledProperty = new PropertyAPI( options.enabledPropertyOptions );
    // }
    //
    // // TODO: not supported yet, see https://github.com/phetsims/scenery/issues/1098
    // if ( options.opacityPropertyInstrumented ) {
    //
    //   // @public (read-only)
    //   this.opacityProperty = new PropertyAPI( options.opacityPropertyOptions );
    // }
  }
}

scenery.register( 'NodeSpecification', NodeSpecification );
export default NodeSpecification;