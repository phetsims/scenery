// Copyright 2024, University of Colorado Boulder

/**
 * The GatedVisibleProperty class abstracts the process of creating a "gated" visibility Property
 * designed for PhET-iO integration. This pattern comes in handy when an object's visibility is already controlled
 * within the simulation, but there is a need to grant additional visibility control to an external entity,
 * such as a studio or a PhET-iO client.
 *
 * @author Marla Schulz (PhET Interactive Simulations)
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import BooleanProperty, { BooleanPropertyOptions } from '../../../axon/js/BooleanProperty.js';
import { DerivedProperty2, DerivedPropertyOptions } from '../../../axon/js/DerivedProperty.js';
import TProperty from '../../../axon/js/TProperty.js';
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import optionize from '../../../phet-core/js/optionize.js';
import StrictOmit from '../../../phet-core/js/types/StrictOmit.js';
import Tandem from '../../../tandem/js/Tandem.js';
import BooleanIO from '../../../tandem/js/types/BooleanIO.js';
import { scenery } from '../imports.js';

type SelfOptions = {
  selfVisibleInitiallyVisible?: boolean;
  selfVisiblePropertyOptions?: BooleanPropertyOptions;
};
type ParentOptions = DerivedPropertyOptions<boolean>;
type GatedVisiblePropertyOptions = SelfOptions & StrictOmit<ParentOptions, 'tandem'>;

class GatedVisibleProperty extends DerivedProperty2<boolean, boolean, boolean> {
  public readonly selfVisibleProperty: TProperty<boolean>;

  public constructor( providedVisibleProperty: TReadOnlyProperty<boolean>, parentTandem: Tandem, providedOptions?: GatedVisiblePropertyOptions ) {

    const options = optionize<GatedVisiblePropertyOptions, SelfOptions, ParentOptions>()( {
      selfVisibleInitiallyVisible: true,
      selfVisiblePropertyOptions: {
        tandem: parentTandem.createTandem( 'selfVisibleProperty' ),
        phetioFeatured: true,
        phetioDocumentation: 'Provides an additional way to toggle the visibility for the PhET-iO Element.'
      },

      tandem: parentTandem.createTandem( 'visibleProperty' ),
      phetioValueType: BooleanIO
      // see below for phetioDocumentation
    }, providedOptions );

    const selfVisibleProperty = new BooleanProperty( options.selfVisibleInitiallyVisible, options.selfVisiblePropertyOptions );
    if ( !options.phetioDocumentation ) {
      options.phetioDocumentation = `Whether the PhET-iO Element is visible, see ${selfVisibleProperty.tandem.name} for customization.`;
    }

    super(
      [ providedVisibleProperty, selfVisibleProperty ],
      ( providedVisible, selfVisible ) => providedVisible && selfVisible,
      options
    );

    this.selfVisibleProperty = selfVisibleProperty;
  }

  public override dispose(): void {

    // Remove the selfVisibleProperty from the PhET-iO registry
    this.selfVisibleProperty.dispose();
    super.dispose();
  }
}

export default GatedVisibleProperty;

scenery.register( 'GatedVisibleProperty', GatedVisibleProperty );