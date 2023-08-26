// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram to convert Oklab => linear sRGB
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, constantTrue, RenderColor, RenderPath, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default class RenderOklabToLinearSRGB extends RenderProgram {
  public constructor(
    public readonly program: RenderProgram
  ) {
    super();
  }

  public override getName(): string {
    return 'RenderOklabToLinearSRGB';
  }

  public override getChildren(): RenderProgram[] {
    return [ this.program ];
  }

  public override withChildren( children: RenderProgram[] ): RenderOklabToLinearSRGB {
    assert && assert( children.length === 1 );
    return new RenderOklabToLinearSRGB( children[ 0 ] );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return other instanceof RenderOklabToLinearSRGB &&
           this.program.equals( other.program );
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): RenderProgram {
    const program = this.program.simplify( pathTest );

    if ( program.isFullyTransparent() ) {
      return RenderColor.TRANSPARENT;
    }

    // Now we're "inside" our path
    if ( program instanceof RenderColor ) {
      return new RenderColor( RenderColor.oklabToLinear( program.color ) );
    }
    else {
      return new RenderOklabToLinearSRGB( program );
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

    return RenderColor.oklabToLinear( source );
  }

  public override serialize(): SerializedRenderOklabToLinearSRGB {
    return {
      type: 'RenderOklabToLinearSRGB',
      program: this.program.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderOklabToLinearSRGB ): RenderOklabToLinearSRGB {
    return new RenderOklabToLinearSRGB( RenderProgram.deserialize( obj.program ) );
  }
}

scenery.register( 'RenderOklabToLinearSRGB', RenderOklabToLinearSRGB );

export type SerializedRenderOklabToLinearSRGB = {
  type: 'RenderOklabToLinearSRGB';
  program: SerializedRenderProgram;
};
