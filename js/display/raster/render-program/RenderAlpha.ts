// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for alpha (an opacity) applied to a RenderProgram
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, constantTrue, RenderColor, RenderPath, RenderProgram, RenderUnary, scenery, SerializedRenderProgram } from '../../../imports.js';
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

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): RenderProgram {
    const program = this.program.simplify( pathTest );

    if ( program.isFullyTransparent() || this.alpha === 0 ) {
      return RenderColor.TRANSPARENT;
    }

    if ( this.alpha === 1 ) {
      return program;
    }

    // Now we're "inside" our path
    if ( program instanceof RenderColor ) {
      return new RenderColor( program.color.timesScalar( this.alpha ) );
    }
    else {
      return new RenderAlpha( program, this.alpha );
    }
  }

  public override evaluate(
    face: ClippableFace | null,
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number,
    pathTest: ( renderPath: RenderPath ) => boolean = constantTrue
  ): Vector4 {
    const source = this.program.evaluate( face, area, centroid, minX, minY, maxX, maxY, pathTest );

    return source.timesScalar( this.alpha );
  }

  protected override getExtraDebugString(): string {
    return `${this.alpha}`;
  }

  public override serialize(): SerializedRenderAlpha {
    return {
      type: 'RenderAlpha',
      program: this.program.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderAlpha ): RenderAlpha {
    return new RenderAlpha( RenderProgram.deserialize( obj.program ), 1 );
  }
}

scenery.register( 'RenderAlpha', RenderAlpha );

export type SerializedRenderAlpha = {
  type: 'RenderAlpha';
  program: SerializedRenderProgram;
};
