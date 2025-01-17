// Copyright 2022-2025, University of Colorado Boulder

/**
 * VBox is a convenience specialization of FlowBox with vertical orientation.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid (PhET Interactive Simulations)
 */

import optionize, { EmptySelfOptions } from '../../../../phet-core/js/optionize.js';
import StrictOmit from '../../../../phet-core/js/types/StrictOmit.js';
import { FlowBox, FlowBoxOptions, Node, scenery, VSeparator } from '../../imports.js';

type SelfOptions = EmptySelfOptions;
export type VBoxOptions = StrictOmit<FlowBoxOptions, 'orientation'>;

export default class VBox extends FlowBox {
  public constructor( providedOptions?: VBoxOptions ) {
    assert && assert( !providedOptions || !( providedOptions as FlowBoxOptions ).orientation, 'VBox sets orientation' );

    super( optionize<VBoxOptions, SelfOptions, FlowBoxOptions>()( {
      orientation: 'vertical'
    }, providedOptions ) );
  }

  protected override onFlowBoxChildInserted( node: Node, index: number ): void {
    assert && assert( !( node instanceof VSeparator ), 'VSeparator should not be used in an VBox. Use HSeparator instead' );

    super.onFlowBoxChildInserted( node, index );
  }

  public override mutate( options?: VBoxOptions ): this {
    return super.mutate( options );
  }
}

scenery.register( 'VBox', VBox );