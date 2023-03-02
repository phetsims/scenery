// Copyright 2022-2023, University of Colorado Boulder

/**
 * HBox is a convenience specialization of FlowBox with horizontal orientation.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import optionize, { EmptySelfOptions } from '../../../../phet-core/js/optionize.js';
import StrictOmit from '../../../../phet-core/js/types/StrictOmit.js';
import { FlowBox, FlowBoxOptions, scenery, Node, HSeparator } from '../../imports.js';

export type HBoxOptions = StrictOmit<FlowBoxOptions, 'orientation'>;

export default class HBox extends FlowBox {
  public constructor( providedOptions?: HBoxOptions ) {
    assert && assert( !providedOptions || !( providedOptions as FlowBoxOptions ).orientation, 'HBox sets orientation' );

    super( optionize<HBoxOptions, EmptySelfOptions, FlowBoxOptions>()( {
      orientation: 'horizontal'
    }, providedOptions ) );
  }

  protected override onFlowBoxChildInserted( node: Node, index: number ): void {
    assert && assert( !( node instanceof HSeparator ), 'HSeparator should not be used in an HBox. Use VSeparator instead' );

    super.onFlowBoxChildInserted( node, index );
  }

  public override mutate( options?: HBoxOptions ): this {
    return super.mutate( options );
  }
}

scenery.register( 'HBox', HBox );
