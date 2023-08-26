// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram to unpremultiply the input color
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, constantTrue, RenderColor, RenderPath, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default class RenderUnpremultiply extends RenderProgram {
  public constructor(
    public readonly program: RenderProgram
  ) {
    super();
  }

  public override getName(): string {
    return 'RenderUnpremultiply';
  }

  public override getChildren(): RenderProgram[] {
    return [ this.program ];
  }

  public override withChildren( children: RenderProgram[] ): RenderUnpremultiply {
    assert && assert( children.length === 1 );
    return new RenderUnpremultiply( children[ 0 ] );
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): RenderProgram {
    const program = this.program.simplify( pathTest );

    if ( program.isFullyTransparent() ) {
      return RenderColor.TRANSPARENT;
    }

    // Now we're "inside" our path
    if ( program instanceof RenderColor ) {
      return new RenderColor( RenderColor.unpremultiply( program.color ) );
    }
    else {
      return new RenderUnpremultiply( program );
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

    return RenderColor.unpremultiply( source );
  }

  public override serialize(): SerializedRenderUnpremultiply {
    return {
      type: 'RenderUnpremultiply',
      program: this.program.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderUnpremultiply ): RenderUnpremultiply {
    return new RenderUnpremultiply( RenderProgram.deserialize( obj.program ) );
  }
}

scenery.register( 'RenderUnpremultiply', RenderUnpremultiply );

export type SerializedRenderUnpremultiply = {
  type: 'RenderUnpremultiply';
  program: SerializedRenderProgram;
};
