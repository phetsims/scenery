// Copyright 2022-2025, University of Colorado Boulder

/**
 * HBox is a convenience specialization of FlowBox with horizontal orientation.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid (PhET Interactive Simulations)
 */

import optionize, { EmptySelfOptions } from '../../../../phet-core/js/optionize.js';
import StrictOmit from '../../../../phet-core/js/types/StrictOmit.js';
import FlowBox from '../../layout/nodes/FlowBox.js';
import type { FlowBoxOptions } from '../../layout/nodes/FlowBox.js';
import HSeparator from '../../layout/nodes/HSeparator.js';
import Node from '../../nodes/Node.js';
import scenery from '../../scenery.js';

type SelfOptions = EmptySelfOptions;
export type HBoxOptions = StrictOmit<FlowBoxOptions, 'orientation'>;

export default class HBox extends FlowBox {
  public constructor( providedOptions?: HBoxOptions ) {
    assert && assert( !providedOptions || !( providedOptions as FlowBoxOptions ).orientation, 'HBox sets orientation' );

    super( optionize<HBoxOptions, SelfOptions, FlowBoxOptions>()( {
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