// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for alpha (an opacity) applied to a RenderProgram
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { RenderColor, RenderEvaluationContext, RenderInstruction, RenderInstructionMultiplyScalar, RenderPathBoolean, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default class RenderAlpha extends RenderProgram {
  public constructor(
    public readonly program: RenderProgram,
    public readonly alpha: number
  ) {
    super(
      [ program ],
      alpha === 0 || program.isFullyTransparent,
      alpha === 1 && program.isFullyOpaque
    );
  }

  public override getName(): string {
    return 'RenderAlpha';
  }

  public override withChildren( children: RenderProgram[] ): RenderAlpha {
    assert && assert( children.length === 1 );
    return new RenderAlpha( children[ 0 ], this.alpha );
  }

  protected override equalsTyped( other: this ): boolean {
    return this.alpha === other.alpha;
  }

  public override getSimplified( children: RenderProgram[] ): RenderProgram | null {
    const program = children[ 0 ];

    if ( program.isFullyTransparent || this.alpha === 0 ) {
      return RenderColor.TRANSPARENT;
    }
    else if ( this.alpha === 1 ) {
      return program;
    }
    else if ( program instanceof RenderColor ) {
      return new RenderColor( program.color.timesScalar( this.alpha ) );
    }
    // Move our path-booleans "up" to the top level (so we can combine other things AND improve path-boolean replacement performance)
    else if ( program instanceof RenderPathBoolean && program.isOneSided() ) {
      return program.withOneSide( this.withChildren( [ program.getOneSide() ] ) ).simplified();
    }
    else {
      return null;
    }
  }

  public override evaluate( context: RenderEvaluationContext ): Vector4 {
    const source = this.program.evaluate( context );

    return source.timesScalar( this.alpha );
  }

  public override writeInstructions( instructions: RenderInstruction[] ): void {
    this.program.writeInstructions( instructions );
    instructions.push( new RenderInstructionMultiplyScalar( this.alpha ) );
  }

  protected override getExtraDebugString(): string {
    return `${this.alpha}`;
  }

  public override serialize(): SerializedRenderAlpha {
    return {
      type: 'RenderAlpha',
      program: this.program.serialize(),
      alpha: this.alpha
    };
  }

  public static override deserialize( obj: SerializedRenderAlpha ): RenderAlpha {
    return new RenderAlpha( RenderProgram.deserialize( obj.program ), obj.alpha );
  }
}

scenery.register( 'RenderAlpha', RenderAlpha );

export type SerializedRenderAlpha = {
  type: 'RenderAlpha';
  program: SerializedRenderProgram;
  alpha: number;
};
