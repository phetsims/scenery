// Copyright 2023, University of Colorado Boulder

/**
 * Function that factors out the creation of a gated visible property for PhET-iO. This is used when we want to grant
 * visibility control to a studio or PhET-iO client to an object that already has its visibility controlled by the sim.
 *
 * @author Marla Schulz (PhET Interactive Simulations)
 *
 */

import Tandem from '../../../../tandem/js/Tandem.js';
import DerivedProperty from '../../../../axon/js/DerivedProperty.js';
import BooleanProperty from '../../../../axon/js/BooleanProperty.js';
import BooleanIO from '../../../../tandem/js/types/BooleanIO.js';
import TReadOnlyProperty from '../../../../axon/js/TReadOnlyProperty.js';
import centerAndVariability from '../../centerAndVariability.js';
import { PhetioObjectOptions } from '../../../../tandem/js/PhetioObject.js';
import { combineOptions } from '../../../../phet-core/js/optionize.js';

export const createGatedVisibleProperty = ( visibleProperty: TReadOnlyProperty<boolean>, tandem: Tandem, selfVisiblePropertyOptions?: PhetioObjectOptions ): TReadOnlyProperty<boolean> => {
  return DerivedProperty.and( [ visibleProperty, new BooleanProperty( true, combineOptions<PhetioObjectOptions>( {
    tandem: tandem.createTandem( 'selfVisibleProperty' ),
    phetioFeatured: true
  }, selfVisiblePropertyOptions ) ) ], {
    tandem: tandem.createTandem( 'visibleProperty' ),
    phetioValueType: BooleanIO
  } );
};

centerAndVariability.register( 'createGatedVisibleProperty', createGatedVisibleProperty );