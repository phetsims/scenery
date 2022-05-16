// Copyright 2022, University of Colorado Boulder

/**
 * VBox is a convenience specialization of LayoutBox with vertical orientation.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import optionize from '../../../../phet-core/js/optionize.js';
import { scenery, LayoutBox, LayoutBoxOptions } from '../../imports.js';

export type VBoxOptions = Omit<LayoutBoxOptions, 'orientation'>;

export default class VBox extends LayoutBox {
  constructor( providedOptions?: VBoxOptions ) {
    assert && assert( !providedOptions || !( providedOptions as LayoutBoxOptions ).orientation, 'VBox sets orientation' );

    super( optionize<VBoxOptions, {}, LayoutBoxOptions>()( {
      orientation: 'vertical'
    }, providedOptions ) );
  }
}

scenery.register( 'VBox', VBox );
