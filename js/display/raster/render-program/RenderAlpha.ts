// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for alpha (an opacity) applied to a RenderProgram
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderColor, RenderProgram, RenderUnary, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default class RenderAlpha extends RenderUnary {
  public constructor(
    program: RenderProgram,
    public readonly alpha: number
  ) {
    super( program );
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

  public override isFullyTransparent(): boolean {
    return this.alpha === 0 || super.isFullyTransparent();
  }

  public override isFullyOpaque(): boolean {
    return this.alpha === 1 && super.isFullyOpaque();
  }

  public override simplified(): RenderProgram {
    const program = this.program.simplified();

    if ( program.isFullyTransparent() || this.alpha === 0 ) {
      return RenderColor.TRANSPARENT;
    }

    if ( this.alpha === 1 ) {
      return program;
    }

    if ( program instanceof RenderColor ) {
      return new RenderColor( program.color.timesScalar( this.alpha ) );
    }
    else if ( program !== this.program ) {
      return new RenderAlpha( program, this.alpha );
    }
    else {
      return this;
    }
  }

  public override evaluate(
    face: ClippableFace | null,
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): Vector4 {
    const source = this.program.evaluate( face, area, centroid, minX, minY, maxX, maxY );

    return source.timesScalar( this.alpha );
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
