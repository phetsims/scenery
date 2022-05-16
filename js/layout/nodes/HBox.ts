// Copyright 2022, University of Colorado Boulder

/**
 * HBox is a convenience specialization of LayoutBox with horizontal orientation.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import optionize from '../../../../phet-core/js/optionize.js';
import { scenery, LayoutBox, LayoutBoxOptions } from '../../imports.js';

export type HBoxOptions = Omit<LayoutBoxOptions, 'orientation'>;

export default class HBox extends LayoutBox {
  constructor( providedOptions?: HBoxOptions ) {
    assert && assert( !providedOptions || !( providedOptions as LayoutBoxOptions ).orientation, 'HBox sets orientation' );

    super( optionize<HBoxOptions, {}, LayoutBoxOptions>()( {
      orientation: 'horizontal'
    }, providedOptions ) );
  }
}

scenery.register( 'HBox', HBox );
