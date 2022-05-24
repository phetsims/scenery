// Copyright 2022, University of Colorado Boulder

/**
 * HBox is a convenience specialization of FlowBox with horizontal orientation.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import optionize from '../../../../phet-core/js/optionize.js';
import { scenery, FlowBox, FlowBoxOptions } from '../../imports.js';

export type HBoxOptions = Omit<FlowBoxOptions, 'orientation'>;

export default class HBox extends FlowBox {
  constructor( providedOptions?: HBoxOptions ) {
    assert && assert( !providedOptions || !( providedOptions as FlowBoxOptions ).orientation, 'HBox sets orientation' );

    super( optionize<HBoxOptions, {}, FlowBoxOptions>()( {
      orientation: 'horizontal'
    }, providedOptions ) );
  }
}

scenery.register( 'HBox', HBox );
