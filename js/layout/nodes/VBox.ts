// Copyright 2022, University of Colorado Boulder

/**
 * VBox is a convenience specialization of FlowBox with vertical orientation.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import optionize from '../../../../phet-core/js/optionize.js';
import { scenery, FlowBox, FlowBoxOptions } from '../../imports.js';

export type VBoxOptions = Omit<FlowBoxOptions, 'orientation'>;

export default class VBox extends FlowBox {
  constructor( providedOptions?: VBoxOptions ) {
    assert && assert( !providedOptions || !( providedOptions as FlowBoxOptions ).orientation, 'VBox sets orientation' );

    super( optionize<VBoxOptions, {}, FlowBoxOptions>()( {
      orientation: 'vertical'
    }, providedOptions ) );
  }
}

scenery.register( 'VBox', VBox );
