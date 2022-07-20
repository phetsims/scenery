// Copyright 2022, University of Colorado Boulder

/**
 * VBox is a convenience specialization of FlowBox with vertical orientation.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import optionize, { EmptySelfOptions } from '../../../../phet-core/js/optionize.js';
import StrictOmit from '../../../../phet-core/js/types/StrictOmit.js';
import { FlowBox, FlowBoxOptions, scenery } from '../../imports.js';

export type VBoxOptions = StrictOmit<FlowBoxOptions, 'orientation'>;

export default class VBox extends FlowBox {
  public constructor( providedOptions?: VBoxOptions ) {
    assert && assert( !providedOptions || !( providedOptions as FlowBoxOptions ).orientation, 'VBox sets orientation' );

    super( optionize<VBoxOptions, EmptySelfOptions, FlowBoxOptions>()( {
      orientation: 'vertical'
    }, providedOptions ) );
  }
}

scenery.register( 'VBox', VBox );
